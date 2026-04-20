"""DreamerV3 training on StallRecoveryEnv.

    uv run train                                       # quick profile (~1.5-2 hr)
    uv run train --profile good                        # good profile (~7-8 hr)
    uv run train --profile best                        # best profile (~35-40 hr)
    uv run train --run-name baseline                   # custom output dir name
    uv run train --resume-from runs/dreamer/prior --run-name continued

Outputs land under ``runs/dreamer/<run-name>/``: TensorBoard events,
``latest.pt`` (saved every 10k steps + on Ctrl+C), ``best.pt`` (saved when the
fixed no-curriculum validation suite improves), ``config.yaml`` and
``env_config.yaml`` (for play/evaluate to rebuild the exact run environment).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from tqdm import tqdm

# Vendored r2dreamer must be on sys.path *before* its modules are imported
# anywhere (including in subprocess workers spawned by ParallelEnv).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "r2dreamer"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from dreamliner.envs import DreamerStallEnv  # noqa: E402
from dreamliner.utils import jsbsim_utils as J  # noqa: E402

# r2dreamer modules (loaded after sys.path mutation).
import tools  # noqa: E402
from buffer import Buffer  # noqa: E402
from dreamer import Dreamer  # noqa: E402
from envs.parallel import ParallelEnv  # noqa: E402
from trainer import OnlineTrainer  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")

log = logging.getLogger("dreamer.train")
_ACTION_NAMES = ("elevator", "aileron", "rudder", "throttle")
_ACTION_SATURATION_THRESHOLD = 0.95


# ----------------------------------------------------------------------------
# Training profiles
# ----------------------------------------------------------------------------
# Timings measured on an RTX 5090. Every knob lives here so the CLI stays tiny.

PROFILES: dict[str, dict] = {
    "quick": {
        "total_steps":      250_000,
        "model_size":       "100M",
        "action_repeat":    1,
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
        "model_size":       "100M",
        "action_repeat":    1,
        "num_envs":         32,
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
    "best": {
        "total_steps":      5_000_000,
        "model_size":       "100M",
        "action_repeat":    1,
        "num_envs":         32,
        "eval_episodes":    16,
        "batch_size":       64,
        "batch_length":     64,
        "train_ratio":      512,
        "buffer_size":      5e5,
        "eval_every":       50_000,
        "log_every":        25_000,
        "checkpoint_every": 50_000,
        "time_limit":       600,
    },
}

_DEVICE = "cuda:0"
_SEED = 0
_VALIDATION_EPISODES_MIN = 36
_VALIDATION_SEED = 12345


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
# MSVC environment setup (Windows only)
# ----------------------------------------------------------------------------
# torch.compile's Inductor C++ codegen shells out to cl.exe when it falls off
# the Triton-only fast path. Windows shipped VS Build Tools doesn't put cl.exe
# on PATH unless you launch from a Developer Command Prompt, which defeats
# "double-click train and walk away". We run vcvars64.bat once at startup and
# graft its env into our process so torch.compile just works from plain
# PowerShell. No-op on Linux/Mac.

def _find_vcvars64() -> Path | None:
    vswhere = Path(r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe")
    if vswhere.exists():
        try:
            out = subprocess.check_output(
                [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"],
                text=True,
            ).strip()
            if out:
                candidate = Path(out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if candidate.exists():
                    return candidate
        except (subprocess.CalledProcessError, OSError):
            pass
    for root in (r"C:\Program Files (x86)\Microsoft Visual Studio",
                 r"C:\Program Files\Microsoft Visual Studio"):
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            p = Path(root) / "2022" / edition / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if p.exists():
                return p
    return None


def _ensure_msvc_env_on_windows() -> None:
    if sys.platform != "win32":
        return
    if shutil.which("cl.exe"):
        log.info("MSVC cl.exe already on PATH; skipping vcvars64 setup.")
        return
    vcvars = _find_vcvars64()
    if vcvars is None:
        log.warning(
            "MSVC cl.exe not on PATH and no VS 2022 install found. torch.compile "
            "may fall back to eager for some ops. Install VS 2022 Build Tools to fix."
        )
        return
    try:
        proc = subprocess.run(
            f'"{vcvars}" >nul && set',
            shell=True, capture_output=True, text=True, check=True,
            encoding="mbcs", errors="replace",
        )
    except subprocess.CalledProcessError as e:
        log.warning("vcvars64.bat failed (rc=%s); torch.compile may degrade.", e.returncode)
        return
    updated = 0
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        if os.environ.get(key) != value:
            os.environ[key] = value
            updated += 1
    log.info("Loaded MSVC env from %s (%d vars updated)", vcvars, updated)


# ----------------------------------------------------------------------------
# Env factory (module-level so it survives subprocess spawn on Windows).
# ----------------------------------------------------------------------------

def _make_env(
    seed: int,
    env_config_path: str | None = None,
    curriculum_step_file: str | None = None,
    disable_curriculum: bool = False,
) -> DreamerStallEnv:
    return DreamerStallEnv(
        seed=seed,
        config=env_config_path,
        curriculum_step_file=curriculum_step_file,
        disable_curriculum=disable_curriculum,
    )


def _env_constructor(
    base_seed: int,
    env_config_path: str | None,
    curriculum_step_file: str | None,
    disable_curriculum: bool,
    idx: int,
):
    return functools.partial(
        _make_env,
        seed=base_seed + idx,
        env_config_path=env_config_path,
        curriculum_step_file=curriculum_step_file,
        disable_curriculum=disable_curriculum,
    )


def _obs_to_trans(obs: dict[str, np.ndarray], device: str) -> TensorDict:
    tensors = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0) for k, v in obs.items()}
    td = TensorDict(tensors, batch_size=(1,))
    for key in td.keys():
        if td[key].ndim == 1:
            td[key] = td[key].unsqueeze(-1)
    return td.to(device, non_blocking=True)


def _make_action_stats() -> dict[str, Any]:
    dims = len(_ACTION_NAMES)
    return {
        "count": 0,
        "sum": np.zeros(dims, dtype=np.float64),
        "sq_sum": np.zeros(dims, dtype=np.float64),
        "abs_sum": np.zeros(dims, dtype=np.float64),
        "min": np.full(dims, np.inf, dtype=np.float64),
        "max": np.full(dims, -np.inf, dtype=np.float64),
        "sat_count": np.zeros(dims, dtype=np.float64),
        "any_sat_count": 0.0,
    }


def _update_action_stats(stats: dict[str, Any], action: np.ndarray) -> None:
    action = np.asarray(action, dtype=np.float64).reshape(len(_ACTION_NAMES))
    stats["count"] += 1
    stats["sum"] += action
    stats["sq_sum"] += action * action
    stats["abs_sum"] += np.abs(action)
    stats["min"] = np.minimum(stats["min"], action)
    stats["max"] = np.maximum(stats["max"], action)
    saturated = np.abs(action) >= _ACTION_SATURATION_THRESHOLD
    stats["sat_count"] += saturated.astype(np.float64)
    stats["any_sat_count"] += float(np.any(saturated))


def _finalize_action_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats["count"])
    if count <= 0:
        return {
            "steps": 0,
            "any_saturation_frac": 0.0,
            **{
                name: {
                    "mean": 0.0,
                    "std": 0.0,
                    "abs_mean": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "sat_frac": 0.0,
                }
                for name in _ACTION_NAMES
            },
        }

    result = {
        "steps": count,
        "any_saturation_frac": float(stats["any_sat_count"] / count),
    }
    mean_vec = stats["sum"] / count
    sq_mean_vec = stats["sq_sum"] / count
    std_vec = np.sqrt(np.maximum(0.0, sq_mean_vec - mean_vec * mean_vec))
    abs_mean_vec = stats["abs_sum"] / count
    sat_frac_vec = stats["sat_count"] / count
    for idx, name in enumerate(_ACTION_NAMES):
        result[name] = {
            "mean": float(mean_vec[idx]),
            "std": float(std_vec[idx]),
            "abs_mean": float(abs_mean_vec[idx]),
            "min": float(stats["min"][idx]),
            "max": float(stats["max"][idx]),
            "sat_frac": float(sat_frac_vec[idx]),
        }
    return result


def _merge_action_stats(dst: dict[str, Any], src: dict[str, Any]) -> None:
    if src["count"] <= 0:
        return
    if dst["count"] <= 0:
        dst["min"] = src["min"].copy()
        dst["max"] = src["max"].copy()
    else:
        dst["min"] = np.minimum(dst["min"], src["min"])
        dst["max"] = np.maximum(dst["max"], src["max"])
    dst["count"] += src["count"]
    dst["sum"] += src["sum"]
    dst["sq_sum"] += src["sq_sum"]
    dst["abs_sum"] += src["abs_sum"]
    dst["sat_count"] += src["sat_count"]
    dst["any_sat_count"] += src["any_sat_count"]


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _curriculum_scale(total_steps: int) -> int:
    quick_steps = int(PROFILES["quick"]["total_steps"])
    return max(1, int(round(total_steps / quick_steps)))


def _scale_env_curriculum(env_config: dict, scale: int) -> dict:
    if scale == 1:
        return env_config
    scaled = deepcopy(env_config)
    phases = scaled.get("curriculum", {}).get("phases", [])
    for phase in phases:
        start_step = int(phase.get("start_step", 0))
        if start_step > 0:
            phase["start_step"] = int(round(start_step * scale))
    return scaled


def _prepare_env_config(
    logdir: Path,
    profile_name: str,
    profile: dict,
    resume_from: str | None,
) -> Path:
    out_path = logdir / "env_config.yaml"
    default_env_config = _load_yaml(J.default_config_path())
    if resume_from:
        resume_snapshot = Path(resume_from) / "env_config.yaml"
        if resume_snapshot.exists():
            shutil.copy2(resume_snapshot, out_path)
            log.info("Copied env config snapshot from %s", resume_snapshot)
            return out_path
        log.warning(
            "Resume source %s has no env_config.yaml snapshot; using current default.yaml unchanged.",
            resume_from,
        )
        _save_yaml(out_path, default_env_config)
        return out_path

    scale = _curriculum_scale(int(profile["total_steps"]))
    env_config = _scale_env_curriculum(default_env_config, scale)
    if scale != 1:
        log.info(
            "Scaled curriculum phase boundaries by %dx for profile %s.",
            scale,
            profile_name,
        )
    _save_yaml(out_path, env_config)
    return out_path


def _run_fixed_validation(
    agent,
    env_config_path: Path,
    num_episodes: int,
    device: str,
    seed: int = _VALIDATION_SEED,
) -> dict[str, Any]:
    was_training = agent.training
    agent.train(False)
    env = DreamerStallEnv(seed=seed, config=env_config_path, disable_curriculum=True)
    try:
        env_config = _load_yaml(env_config_path)
        scenario_names = list(env_config["scenarios"].keys())
        if not scenario_names:
            raise ValueError(f"validation config has no scenarios: {env_config_path}")
        episodes_per_scenario = max(1, math.ceil(num_episodes / len(scenario_names)))
        per_scenario_raw = {
            scenario_name: {
                "success": 0,
                "crash": 0,
                "timeout": 0,
                "returns": [],
                "altitude_losses": [],
                "lengths": [],
                "action_stats": _make_action_stats(),
            }
            for scenario_name in scenario_names
        }

        success = 0
        crash = 0
        timeout = 0
        returns: list[float] = []
        altitude_losses: list[float] = []
        lengths: list[int] = []
        action_stats = _make_action_stats()
        episode_records: list[dict[str, Any]] = []
        for scenario_idx, scenario_name in enumerate(scenario_names):
            for episode_idx in range(episodes_per_scenario):
                episode_seed = seed + scenario_idx * 10_000 + episode_idx
                obs = env.reset(seed=episode_seed, options={"scenario": scenario_name})
                reset_info = dict(env.last_reset_info)
                agent_state = agent.get_initial_state(1)
                total_reward = 0.0
                done = False
                steps = 0
                info: dict[str, Any] = {}
                episode_action_stats = _make_action_stats()
                while not done:
                    trans = _obs_to_trans(obs, device)
                    with torch.no_grad():
                        action, agent_state = agent.act(trans, agent_state, True)
                    action_np = action[0].detach().cpu().numpy().astype(np.float32)
                    _update_action_stats(action_stats, action_np)
                    _update_action_stats(episode_action_stats, action_np)
                    obs, reward, done, info = env.step(action_np)
                    total_reward += float(reward)
                    steps += 1

                returns.append(total_reward)
                altitude_losses.append(float(info["altitude_loss_ft"]))
                lengths.append(steps)
                scenario_metrics = per_scenario_raw[scenario_name]
                scenario_metrics["returns"].append(total_reward)
                scenario_metrics["altitude_losses"].append(float(info["altitude_loss_ft"]))
                scenario_metrics["lengths"].append(steps)
                _merge_action_stats(scenario_metrics["action_stats"], episode_action_stats)
                if info["crashed"]:
                    crash += 1
                    scenario_metrics["crash"] += 1
                elif info["success"]:
                    success += 1
                    scenario_metrics["success"] += 1
                else:
                    timeout += 1
                    scenario_metrics["timeout"] += 1
                episode_record = {
                    "scenario": scenario_name,
                    "seed": int(episode_seed),
                    "episode_index": len(episode_records),
                    "return": float(total_reward),
                    "length": int(steps),
                    "outcome": str(info["outcome"]),
                    "success": bool(info["success"]),
                    "crash": bool(info["crashed"]),
                    "timeout": bool(not info["success"] and not info["crashed"]),
                    "altitude_loss_ft": float(info["altitude_loss_ft"]),
                    "curriculum_step": int(reset_info.get("curriculum_step", 0)),
                    "ics": dict(reset_info.get("ics", {})),
                    "action": _finalize_action_stats(episode_action_stats),
                }
                for key, value in info.items():
                    if key.startswith("log_"):
                        episode_record[key[4:]] = float(value)
                episode_records.append(episode_record)

        total = len(returns)
        per_scenario = {}
        for scenario_name, scenario_metrics in per_scenario_raw.items():
            scenario_total = len(scenario_metrics["returns"])
            per_scenario[scenario_name] = {
                "episodes": int(scenario_total),
                "success_rate": scenario_metrics["success"] / scenario_total,
                "crash_rate": scenario_metrics["crash"] / scenario_total,
                "timeout_rate": scenario_metrics["timeout"] / scenario_total,
                "mean_return": float(mean(scenario_metrics["returns"])),
                "median_return": float(median(scenario_metrics["returns"])),
                "mean_altitude_loss_ft": float(mean(scenario_metrics["altitude_losses"])),
                "median_altitude_loss_ft": float(median(scenario_metrics["altitude_losses"])),
                "mean_length": float(mean(scenario_metrics["lengths"])),
                "median_length": float(median(scenario_metrics["lengths"])),
                "action": _finalize_action_stats(scenario_metrics["action_stats"]),
            }
        return {
            "episodes": int(total),
            "success_rate": success / total,
            "crash_rate": crash / total,
            "timeout_rate": timeout / total,
            "mean_return": float(mean(returns)),
            "median_return": float(median(returns)),
            "mean_altitude_loss_ft": float(mean(altitude_losses)),
            "median_altitude_loss_ft": float(median(altitude_losses)),
            "mean_length": float(mean(lengths)),
            "action": _finalize_action_stats(action_stats),
            "per_scenario": per_scenario,
            "episode_records": episode_records,
        }
    finally:
        env.close()
        agent.train(was_training)


def _validation_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metrics["success_rate"],
        -metrics["crash_rate"],
        -metrics["median_altitude_loss_ft"],
        metrics["mean_return"],
    )


# ----------------------------------------------------------------------------
# Config assembly
# ----------------------------------------------------------------------------

def build_config(profile: dict, logdir: Path, *, total_steps: int) -> OmegaConf:
    base_model = OmegaConf.load(_VENDOR_ROOT / "configs" / "model" / "_base_.yaml")
    size_model = OmegaConf.load(_VENDOR_ROOT / "configs" / "model" / f"size{profile['model_size']}.yaml")
    model = OmegaConf.merge(base_model, size_model)
    # torch.compile wraps _cal_grad with mode="reduce-overhead" (see vendor/r2dreamer/
    # dreamer.py:161). ~30-50% speedup once warm. Windows needs `triton-windows`
    # (installed via uv) plus MSVC's cl.exe on PATH: if launching from plain
    # PowerShell, run from "x64 Native Tools Command Prompt for VS 2022", or call
    # vcvars64.bat first. First compile of the graph is slow (1-5 min); subsequent
    # runs hit the cache at %USERPROFILE%\.triton\cache.
    model.compile = True

    env = OmegaConf.create({
        "task":             "dreamliner_stall_recovery",
        "steps":            int(total_steps),
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
            "steps":            int(total_steps),
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
            "Profiles (RTX 5090 + Ryzen 9 9950X3D timings):\n"
            "  quick   250k steps, 100M model, 16 envs, ~1.5-2 hr, ~22-25 GB VRAM  (default)\n"
            "  good    1M steps,   100M model, 32 envs, ~7-8 hr,   ~22-25 GB VRAM\n"
            "  best    5M steps,   100M model, 32 envs, ~35-40 hr, ~22-25 GB VRAM\n"
        ),
    )
    p.add_argument("--profile", choices=list(PROFILES), default="quick",
                   help="Training profile (default: quick).")
    p.add_argument("--run-name", type=str, default=None,
                   help="Subdir under runs/dreamer/. Default: run-<timestamp>.")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Prior run dir; restores latest.pt into a new run, keeps the "
                        "global step/curriculum position, and treats the selected "
                        "profile's step budget as an additional training block.")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    _setup_logging()
    _ensure_msvc_env_on_windows()

    profile = PROFILES[args.profile]
    resume_ckpt_path: Path | None = None
    resume_step = 0
    if args.resume_from:
        resume_ckpt_path = Path(args.resume_from) / "latest.pt"
        if not resume_ckpt_path.exists():
            raise FileNotFoundError(f"--resume-from: no latest.pt at {resume_ckpt_path}")
        resume_step = int(torch.load(resume_ckpt_path, map_location="cpu", weights_only=False).get("step", 0))

    run_name = args.run_name or f"run-{int(time.time())}"
    logdir = Path("runs/dreamer") / run_name
    logdir.mkdir(parents=True, exist_ok=True)
    env_config_path = _prepare_env_config(logdir, args.profile, profile, args.resume_from)

    target_total_steps = int(profile["total_steps"]) + resume_step
    config = build_config(profile, logdir, total_steps=target_total_steps)

    # Persist the resolved config so play/evaluate can rebuild the agent.
    OmegaConf.save(config, logdir / "config.yaml")

    tools.set_seed_everywhere(config.seed)
    log.info("Logdir: %s", logdir)
    log.info("Profile: %s  |  %d env steps  |  model %s  |  device %s",
             args.profile, target_total_steps, profile["model_size"], config.device)
    if resume_ckpt_path is not None:
        log.info("Resume source: %s  |  starting step %d", resume_ckpt_path, resume_step)

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
    # Initialized to the resumed global step so continuations stay in the same
    # curriculum phase instead of dropping back to phase 0.
    curriculum_step_file = logdir / "_curriculum_step.txt"
    curriculum_step_file.write_text(str(resume_step), encoding="utf-8")
    curriculum_step_path = str(curriculum_step_file.resolve())
    env_config_value = str(env_config_path.resolve())

    log.info("Spawning %d train envs + %d eval envs...",
             config.env.env_num, config.env.eval_episode_num)
    train_envs = ParallelEnv(
        functools.partial(
            _env_constructor,
            int(config.seed),
            env_config_value,
            curriculum_step_path,
            False,
        ),
        int(config.env.env_num),
        config.device,
    )
    eval_envs = ParallelEnv(
        functools.partial(
            _env_constructor,
            int(config.seed) + 10_000,
            env_config_value,
            None,
            True,
        ),
        int(config.env.eval_episode_num),
        config.device,
    )

    log.info("Building Dreamer agent (size=%s)...", profile["model_size"])
    agent = Dreamer(config.model, train_envs.observation_space, train_envs.action_space).to(config.device)

    if resume_ckpt_path is not None:
        log.info("Resuming policy + optimizer state from %s", resume_ckpt_path)
        state = torch.load(resume_ckpt_path, map_location=config.device, weights_only=False)
        agent.load_state_dict(state["agent_state_dict"])
        if "optims_state_dict" in state:
            tools.recursively_load_optim_state_dict(agent, state["optims_state_dict"])
        if "scheduler_state_dict" in state and hasattr(agent, "_scheduler"):
            agent._scheduler.load_state_dict(state["scheduler_state_dict"])
        if "scaler_state_dict" in state and hasattr(agent, "_scaler"):
            agent._scaler.load_state_dict(state["scaler_state_dict"])

    final_path = logdir / "latest.pt"
    best_path  = logdir / "best.pt"
    best_validation = {"key": None, "step": -1, "metrics": None}
    validation_episodes = max(_VALIDATION_EPISODES_MIN, int(config.env.eval_episode_num))
    validation_jsonl_path = logdir / "validation.jsonl"
    validation_episode_jsonl_path = logdir / "validation_episode_diagnostics.jsonl"

    def _write_ckpt(
        path: Path,
        step: int,
        agent_ref,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "agent_state_dict":  agent_ref.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(agent_ref),
            "step":              int(step),
        }
        if hasattr(agent_ref, "_scheduler"):
            payload["scheduler_state_dict"] = agent_ref._scheduler.state_dict()
        if hasattr(agent_ref, "_scaler"):
            payload["scaler_state_dict"] = agent_ref._scaler.state_dict()
        if score is not None:
            payload["eval_score"] = float(score)
        if metadata:
            payload.update(metadata)
        torch.save(payload, path)

    def save_checkpoint(step: int, agent_ref) -> None:
        _write_ckpt(final_path, step, agent_ref)
        log.info("Checkpoint saved at step %d -> %s", step, final_path)

    best_json_path = logdir / "best.json"

    def on_validation(step: int, _score: float, agent_ref) -> None:
        metrics = _run_fixed_validation(agent_ref, env_config_path, validation_episodes, config.device)
        logger.scalar("validation/episodes", metrics["episodes"])
        logger.scalar("validation/success_rate", metrics["success_rate"])
        logger.scalar("validation/crash_rate", metrics["crash_rate"])
        logger.scalar("validation/timeout_rate", metrics["timeout_rate"])
        logger.scalar("validation/mean_return", metrics["mean_return"])
        logger.scalar("validation/median_return", metrics["median_return"])
        logger.scalar("validation/mean_altitude_loss_ft", metrics["mean_altitude_loss_ft"])
        logger.scalar("validation/median_altitude_loss_ft", metrics["median_altitude_loss_ft"])
        logger.scalar("validation/mean_length", metrics["mean_length"])
        logger.scalar("validation/action/any_saturation_frac", metrics["action"]["any_saturation_frac"])
        logger.scalar("validation/action/steps", metrics["action"]["steps"])
        for action_name, action_metrics in metrics["action"].items():
            if action_name in {"steps", "any_saturation_frac"}:
                continue
            prefix = f"validation/action/{action_name}"
            logger.scalar(f"{prefix}/mean", action_metrics["mean"])
            logger.scalar(f"{prefix}/std", action_metrics["std"])
            logger.scalar(f"{prefix}/abs_mean", action_metrics["abs_mean"])
            logger.scalar(f"{prefix}/min", action_metrics["min"])
            logger.scalar(f"{prefix}/max", action_metrics["max"])
            logger.scalar(f"{prefix}/sat_frac", action_metrics["sat_frac"])
        for scenario_name, scenario_metrics in metrics["per_scenario"].items():
            prefix = f"validation/by_scenario/{scenario_name}"
            logger.scalar(f"{prefix}/episodes", scenario_metrics["episodes"])
            logger.scalar(f"{prefix}/success_rate", scenario_metrics["success_rate"])
            logger.scalar(f"{prefix}/crash_rate", scenario_metrics["crash_rate"])
            logger.scalar(f"{prefix}/timeout_rate", scenario_metrics["timeout_rate"])
            logger.scalar(f"{prefix}/mean_return", scenario_metrics["mean_return"])
            logger.scalar(f"{prefix}/median_return", scenario_metrics["median_return"])
            logger.scalar(
                f"{prefix}/mean_altitude_loss_ft",
                scenario_metrics["mean_altitude_loss_ft"],
            )
            logger.scalar(
                f"{prefix}/median_altitude_loss_ft",
                scenario_metrics["median_altitude_loss_ft"],
            )
            logger.scalar(f"{prefix}/mean_length", scenario_metrics["mean_length"])
            logger.scalar(f"{prefix}/median_length", scenario_metrics["median_length"])
            logger.scalar(
                f"{prefix}/action/any_saturation_frac",
                scenario_metrics["action"]["any_saturation_frac"],
            )
            logger.scalar(f"{prefix}/action/steps", scenario_metrics["action"]["steps"])
            for action_name, action_metrics in scenario_metrics["action"].items():
                if action_name in {"steps", "any_saturation_frac"}:
                    continue
                action_prefix = f"{prefix}/action/{action_name}"
                logger.scalar(f"{action_prefix}/mean", action_metrics["mean"])
                logger.scalar(f"{action_prefix}/std", action_metrics["std"])
                logger.scalar(f"{action_prefix}/abs_mean", action_metrics["abs_mean"])
                logger.scalar(f"{action_prefix}/min", action_metrics["min"])
                logger.scalar(f"{action_prefix}/max", action_metrics["max"])
                logger.scalar(f"{action_prefix}/sat_frac", action_metrics["sat_frac"])

        with validation_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "step": int(step),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "episodes": int(metrics["episodes"]),
                "success_rate": float(metrics["success_rate"]),
                "crash_rate": float(metrics["crash_rate"]),
                "timeout_rate": float(metrics["timeout_rate"]),
                "mean_return": float(metrics["mean_return"]),
                "median_return": float(metrics["median_return"]),
                "mean_altitude_loss_ft": float(metrics["mean_altitude_loss_ft"]),
                "median_altitude_loss_ft": float(metrics["median_altitude_loss_ft"]),
                "mean_length": float(metrics["mean_length"]),
                "action": metrics["action"],
                "per_scenario": metrics["per_scenario"],
            }) + "\n")
        with validation_episode_jsonl_path.open("a", encoding="utf-8") as f:
            timestamp = datetime.now().isoformat(timespec="seconds")
            for record in metrics["episode_records"]:
                f.write(json.dumps({
                    "validation_step": int(step),
                    "timestamp": timestamp,
                    **record,
                }) + "\n")

        key = _validation_key(metrics)
        if best_validation["key"] is None or key > best_validation["key"]:
            prev_metrics = best_validation["metrics"]
            best_validation["key"] = key
            best_validation["step"] = step
            best_validation["metrics"] = metrics
            _write_ckpt(
                best_path,
                step,
                agent_ref,
                score=metrics["mean_return"],
                metadata={
                    "validation_success_rate": metrics["success_rate"],
                    "validation_crash_rate": metrics["crash_rate"],
                    "validation_timeout_rate": metrics["timeout_rate"],
                    "validation_mean_altitude_loss_ft": metrics["mean_altitude_loss_ft"],
                    "validation_median_altitude_loss_ft": metrics["median_altitude_loss_ft"],
                    "validation_mean_length": metrics["mean_length"],
                },
            )
            best_json_path.write_text(json.dumps({
                "step":                           int(step),
                "eval_score":                     float(metrics["mean_return"]),
                "validation_success_rate":        float(metrics["success_rate"]),
                "validation_crash_rate":          float(metrics["crash_rate"]),
                "validation_timeout_rate":        float(metrics["timeout_rate"]),
                "validation_mean_return":         float(metrics["mean_return"]),
                "validation_median_return":       float(metrics["median_return"]),
                "validation_mean_altitude_loss_ft": float(metrics["mean_altitude_loss_ft"]),
                "validation_median_altitude_loss_ft": float(metrics["median_altitude_loss_ft"]),
                "validation_mean_length":         float(metrics["mean_length"]),
                "validation_episodes":            int(metrics["episodes"]),
                "run_name":                       run_name,
                "logdir":                         str(logdir),
                "timestamp":                      datetime.now().isoformat(timespec="seconds"),
                "model_size":                     profile["model_size"],
                "selection_metric":               "success_rate,-crash_rate,-median_altitude_loss_ft,mean_return",
            }, indent=2), encoding="utf-8")
            if prev_metrics is None:
                log.info(
                    "New best validation at step %d -> %s "
                    "(success=%.1f%% crash=%.1f%% median_alt_loss=%.0fft)",
                    step,
                    best_path,
                    metrics["success_rate"] * 100.0,
                    metrics["crash_rate"] * 100.0,
                    metrics["median_altitude_loss_ft"],
                )
            else:
                log.info(
                    "New best validation at step %d -> %s "
                    "(success=%.1f%%, prev %.1f%% | crash=%.1f%%, prev %.1f%% | "
                    "median_alt_loss=%.0fft, prev %.0fft)",
                    step,
                    best_path,
                    metrics["success_rate"] * 100.0,
                    prev_metrics["success_rate"] * 100.0,
                    metrics["crash_rate"] * 100.0,
                    prev_metrics["crash_rate"] * 100.0,
                    metrics["median_altitude_loss_ft"],
                    prev_metrics["median_altitude_loss_ft"],
                )

    pbar = tqdm(
        total=target_total_steps,
        initial=resume_step,
        unit="step",
        smoothing=0.05,
        dynamic_ncols=True,
    )

    # Throttle curriculum step-file writes: ~every 5k steps is finer-grained
    # than the 10k eval cadence, and scenario weights don't need sub-second
    # precision. Atomic replace so env subprocesses never read a half-written file.
    _curriculum_write_every = 5_000
    _last_curriculum_write = {"step": resume_step}

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
        on_eval=on_validation,
    )
    interrupted = False
    try:
        trainer.begin(agent, start_step=resume_step)
    except KeyboardInterrupt:
        interrupted = True
        log.warning("KeyboardInterrupt - flushing final checkpoint before exit.")
    finally:
        pbar.close()

    # Always write a final checkpoint at clean exit OR Ctrl+C.
    final_step = pbar.n if interrupted else trainer.steps
    save_checkpoint(final_step, agent)
    if best_validation["step"] >= 0 and best_validation["metrics"] is not None:
        metrics = best_validation["metrics"]
        log.info(
            "Best validation: success=%.1f%% crash=%.1f%% median_alt_loss=%.0fft "
            "mean_return=%.2f at step %d -> %s",
            metrics["success_rate"] * 100.0,
            metrics["crash_rate"] * 100.0,
            metrics["median_altitude_loss_ft"],
            metrics["mean_return"],
            best_validation["step"],
            best_path,
        )
    if interrupted:
        log.info("Interrupted at step %d. Resume with: --resume-from %s", final_step, logdir)
    log.info("Done. TensorBoard: tensorboard --logdir %s", logdir.parent)


if __name__ == "__main__":
    main()
