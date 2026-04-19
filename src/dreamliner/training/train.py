"""DreamerV3 training on StallRecoveryEnv.

    uv run train                                       # quick profile (~2.7 hr)
    uv run train --profile good                        # good profile (~20 hr)
    uv run train --run-name baseline                   # custom output dir name
    uv run train --resume-from runs/dreamer/prior --run-name continued

Outputs land under ``runs/dreamer/<run-name>/``: TensorBoard events,
``latest.pt`` (saved every 10k steps + on Ctrl+C), ``best.pt`` (saved when
eval score improves), ``config.yaml`` (for play/evaluate to rebuild the agent).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import shutil
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm

# Vendored r2dreamer must be on sys.path *before* its modules are imported
# anywhere (including in subprocess workers spawned by ParallelEnv).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "r2dreamer"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from dreamliner.envs import DreamerStallEnv  # noqa: E402

# r2dreamer modules (loaded after sys.path mutation).
import tools  # noqa: E402
from buffer import Buffer  # noqa: E402
from dreamer import Dreamer  # noqa: E402
from envs.parallel import ParallelEnv  # noqa: E402
from trainer import OnlineTrainer  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

log = logging.getLogger("dreamer.train")


# ----------------------------------------------------------------------------
# Training profiles
# ----------------------------------------------------------------------------
# Timings measured on an RTX 5090. Every knob lives here so the CLI stays tiny.

PROFILES: dict[str, dict] = {
    "quick": {
        "total_steps":      250_000,
        "model_size":       "100M",
        "action_repeat":    2,
        "num_envs":         16,
        "eval_episodes":    16,
        "batch_size":       64,
        "batch_length":     64,
        "train_ratio":      512,
        "buffer_size":      5e5,
        "eval_every":       10_000,
        "log_every":        5_000,
        "checkpoint_every": 10_000,
        "time_limit":       600,
    },
    "good": {
        "total_steps":      1_000_000,
        "model_size":       "200M",
        "action_repeat":    2,
        "num_envs":         16,
        "eval_episodes":    16,
        "batch_size":       64,
        "batch_length":     64,
        "train_ratio":      512,
        "buffer_size":      5e5,
        "eval_every":       10_000,
        "log_every":        5_000,
        "checkpoint_every": 10_000,
        "time_limit":       600,
    },
}

_DEVICE = "cuda:0"
_SEED = 0


class _TqdmLoggingHandler(logging.Handler):
    """Route log records through ``tqdm.write`` so they don't clobber the bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def _setup_logging(level: int = logging.INFO) -> None:
    handler = _TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


# ----------------------------------------------------------------------------
# Env factory (module-level so it survives subprocess spawn on Windows).
# ----------------------------------------------------------------------------

def _make_env(seed: int, curriculum_step_file: str | None = None) -> DreamerStallEnv:
    return DreamerStallEnv(seed=seed, curriculum_step_file=curriculum_step_file)


def _env_constructor(base_seed: int, curriculum_step_file: str | None, idx: int):
    return functools.partial(
        _make_env,
        seed=base_seed + idx,
        curriculum_step_file=curriculum_step_file,
    )


# ----------------------------------------------------------------------------
# Config assembly
# ----------------------------------------------------------------------------

def build_config(profile: dict, logdir: Path) -> OmegaConf:
    base_model = OmegaConf.load(_VENDOR_ROOT / "configs" / "model" / "_base_.yaml")
    size_model = OmegaConf.load(_VENDOR_ROOT / "configs" / "model" / f"size{profile['model_size']}.yaml")
    model = OmegaConf.merge(base_model, size_model)
    model.compile = False  # triton has no Windows wheels; Linux users can edit here.

    env = OmegaConf.create({
        "task":             "dreamliner_stall_recovery",
        "steps":            int(profile["total_steps"]),
        "env_num":          int(profile["num_envs"]),
        "eval_episode_num": int(profile["eval_episodes"]),
        "action_repeat":    int(profile["action_repeat"]),
        "train_ratio":      int(profile["train_ratio"]),
        "time_limit":       int(profile["time_limit"]),
        "size":             [64, 64],   # ignored - no image obs
        "encoder":          {"mlp_keys": "state", "cnn_keys": "$^"},
        "decoder":          {"mlp_keys": "state", "cnn_keys": "$^"},
        "seed":             _SEED,
        "device":           _DEVICE,
    })

    cfg = OmegaConf.create({
        "logdir":            str(logdir),
        "seed":              _SEED,
        "deterministic_run": False,
        "device":            _DEVICE,
        "batch_size":        int(profile["batch_size"]),
        "batch_length":      int(profile["batch_length"]),
        "env":               env,
        "model":             model,
        "buffer": {
            "batch_size":     int(profile["batch_size"]),
            "batch_length":   int(profile["batch_length"]),
            "max_size":       float(profile["buffer_size"]),
            "device":         _DEVICE,
            "storage_device": _DEVICE,
        },
        "trainer": {
            "steps":            int(profile["total_steps"]),
            "pretrain":         0,
            "eval_every":       int(profile["eval_every"]),
            "eval_episode_num": int(profile["eval_episodes"]),
            "batch_size":       int(profile["batch_size"]),
            "batch_length":     int(profile["batch_length"]),
            "train_ratio":      int(profile["train_ratio"]),
            "video_pred_log":   False,
            "params_hist_log":  False,
            "update_log_every": int(profile["log_every"]),
            "action_repeat":    int(profile["action_repeat"]),
        },
    })
    OmegaConf.resolve(cfg)
    return cfg


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DreamerV3 on the 737 stall-recovery env.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Profiles (RTX 5090 timings; action_repeat=2 on both):\n"
            "  quick   250k steps, 100M model, ~2.7 hr, ~22-25 GB VRAM  (default)\n"
            "  good    1M steps,   200M model, ~20 hr,  near-max VRAM\n"
        ),
    )
    p.add_argument("--profile", choices=list(PROFILES), default="quick",
                   help="Training profile (default: quick).")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subdir under runs/dreamer/. Default: run-<timestamp>.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Prior run dir; loads weights from its latest.pt. "
                        "Optimizer + replay buffer are rebuilt fresh.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _setup_logging()

    profile = PROFILES[args.profile]
    run_name = args.run_name or f"run-{int(time.time())}"
    logdir = Path("runs/dreamer") / run_name
    logdir.mkdir(parents=True, exist_ok=True)

    config = build_config(profile, logdir)

    # Persist the resolved config so play/evaluate can rebuild the agent.
    OmegaConf.save(config, logdir / "config.yaml")

    tools.set_seed_everywhere(config.seed)
    log.info("Logdir: %s", logdir)
    log.info("Profile: %s  |  %d env steps  |  model %s  |  device %s",
             args.profile, profile["total_steps"], profile["model_size"], config.device)

    # Disk-space sanity check: TB events + checkpoints + replay snapshots add up.
    free_gb = shutil.disk_usage(logdir.resolve().anchor).free / (1024 ** 3)
    if free_gb < 10:
        log.warning("Low disk space on %s: %.1f GB free. Consider cleaning runs/.",
                    logdir.resolve().anchor, free_gb)
    else:
        log.info("Disk: %.1f GB free on %s", free_gb, logdir.resolve().anchor)

    logger = tools.Logger(logdir)
    logger.log_hydra_config(config)

    log.info("Building replay buffer (max %s, on %s)...",
             int(config.buffer.max_size), config.buffer.storage_device)
    replay_buffer = Buffer(config.buffer)

    # Shared curriculum step file: trainer writes it periodically, every env
    # subprocess reads it on reset() to look up the active curriculum phase.
    # Initialized to 0 so envs spawned before the first update see phase 0.
    curriculum_step_file = logdir / "_curriculum_step.txt"
    curriculum_step_file.write_text("0", encoding="utf-8")
    curriculum_step_path = str(curriculum_step_file.resolve())

    log.info("Spawning %d train envs + %d eval envs...",
             config.env.env_num, config.env.eval_episode_num)
    train_envs = ParallelEnv(
        functools.partial(_env_constructor, int(config.seed), curriculum_step_path),
        int(config.env.env_num),
        config.device,
    )
    eval_envs = ParallelEnv(
        functools.partial(_env_constructor, int(config.seed) + 10_000, curriculum_step_path),
        int(config.env.eval_episode_num),
        config.device,
    )

    log.info("Building Dreamer agent (size=%s)...", profile["model_size"])
    agent = Dreamer(config.model, train_envs.observation_space, train_envs.action_space).to(config.device)

    if args.resume_from:
        ckpt_path = Path(args.resume_from) / "latest.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"--resume-from: no latest.pt at {ckpt_path}")
        log.info("Resuming policy weights from %s", ckpt_path)
        state = torch.load(ckpt_path, map_location=config.device, weights_only=False)
        agent.load_state_dict(state["agent_state_dict"])

    final_path = logdir / "latest.pt"
    best_path  = logdir / "best.pt"
    best_score = {"value": float("-inf"), "step": -1}

    def _write_ckpt(path: Path, step: int, agent_ref, score: float | None = None) -> None:
        payload = {
            "agent_state_dict":  agent_ref.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent_ref),
            "step":              int(step),
        }
        if score is not None:
            payload["eval_score"] = float(score)
        torch.save(payload, path)

    def save_checkpoint(step: int, agent_ref) -> None:
        _write_ckpt(final_path, step, agent_ref)
        log.info("Checkpoint saved at step %d -> %s", step, final_path)

    best_json_path = logdir / "best.json"

    def on_eval_better(step: int, score: float, agent_ref) -> None:
        if score > best_score["value"]:
            prev = best_score["value"]
            best_score["value"] = score
            best_score["step"] = step
            _write_ckpt(best_path, step, agent_ref, score=score)
            best_json_path.write_text(json.dumps({
                "step":       int(step),
                "eval_score": float(score),
                "run_name":   run_name,
                "logdir":     str(logdir),
                "timestamp":  datetime.now().isoformat(timespec="seconds"),
                "model_size": profile["model_size"],
            }, indent=2), encoding="utf-8")
            if prev == float("-inf"):
                log.info("New best (%.2f) at step %d -> %s", score, step, best_path)
            else:
                log.info("New best (%.2f, prev %.2f) at step %d -> %s",
                         score, prev, step, best_path)

    pbar = tqdm(
        total=int(profile["total_steps"]),
        unit="step",
        smoothing=0.05,
        dynamic_ncols=True,
    )

    # Throttle curriculum step-file writes: ~every 5k steps is finer-grained
    # than the 10k eval cadence, and scenario weights don't need sub-second
    # precision. Atomic replace so env subprocesses never read a half-written file.
    _curriculum_write_every = 5_000
    _last_curriculum_write = {"step": -1}

    def update_progress(step: int) -> None:
        delta = step - pbar.n
        if delta > 0:
            pbar.update(delta)
        if step - _last_curriculum_write["step"] >= _curriculum_write_every:
            tmp = curriculum_step_file.with_suffix(".tmp")
            tmp.write_text(str(int(step)), encoding="utf-8")
            tmp.replace(curriculum_step_file)
            _last_curriculum_write["step"] = step

    trainer = OnlineTrainer(
        config.trainer, replay_buffer, logger, logdir, train_envs, eval_envs,
        save_fn=save_checkpoint,
        save_every=profile["checkpoint_every"],
        progress_fn=update_progress,
        on_eval=on_eval_better,
    )
    interrupted = False
    try:
        trainer.begin(agent)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("KeyboardInterrupt - flushing final checkpoint before exit.")
    finally:
        pbar.close()

    # Always write a final checkpoint at clean exit OR Ctrl+C.
    final_step = pbar.n if interrupted else trainer.steps
    save_checkpoint(final_step, agent)
    if best_score["step"] >= 0:
        log.info("Best eval: %.2f at step %d -> %s",
                 best_score["value"], best_score["step"], best_path)
    if interrupted:
        log.info("Interrupted at step %d. Resume with: --resume-from %s", final_step, logdir)
    log.info("Done. TensorBoard: tensorboard --logdir %s", logdir.parent)


if __name__ == "__main__":
    main()
