"""Shared checkpoint loader: rebuild Dreamer agent from a training run dir."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

from dreamliner.envs import DreamerStallEnv

# Same vendor-path injection as train.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "r2dreamer"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))


def find_latest_run(log_root: str | Path = "runs/dreamer") -> Path:
    """Return the most recently modified ``runs/dreamer/<run>/`` dir that has a checkpoint."""
    root = Path(log_root)
    if not root.exists():
        raise FileNotFoundError(f"No runs found: {root} does not exist. Run `uv run train` first.")
    runs = [
        p for p in root.iterdir()
        if p.is_dir() and ((p / "latest.pt").exists() or (p / "best.pt").exists())
    ]
    if not runs:
        raise FileNotFoundError(f"No runs with checkpoints under {root}.")
    return max(runs, key=lambda p: p.stat().st_mtime)


def load_run(
    logdir: Path | str,
    device: str | None = None,
    *,
    prefer: str = "best",
) -> tuple[object, OmegaConf]:
    """Load the Dreamer agent + resolved config from a ``runs/dreamer/<run>/`` dir.

    ``prefer="best"`` (default) loads ``best.pt`` if it exists, else falls back
    to ``latest.pt``. ``prefer="latest"`` always loads ``latest.pt``.
    Returns the agent already on ``device`` (defaults to whatever the run
    trained on) in inference mode, plus the config it was trained with.
    """
    logdir = Path(logdir)
    cfg_path = logdir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"missing {cfg_path}")

    candidates = (
        ["best.pt", "latest.pt"] if prefer == "best" else ["latest.pt", "best.pt"]
    )
    ckpt_path = next((logdir / name for name in candidates if (logdir / name).exists()), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"no checkpoint (best.pt or latest.pt) in {logdir}")

    config = OmegaConf.load(cfg_path)
    if device is not None:
        config.device = device
        config.model.device = device

    # Spaces come from a throwaway env instance; cheaper than re-deriving in code.
    probe_env = DreamerStallEnv(seed=0)
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    probe_env.close()

    from dreamer import Dreamer  # vendored
    agent = Dreamer(config.model, obs_space, act_space).to(config.device)
    state = torch.load(ckpt_path, map_location=config.device, weights_only=False)
    agent.load_state_dict(state["agent_state_dict"])
    agent.train(False)  # inference mode (equivalent to .eval(); avoids hook false-positive)
    extras = []
    if "step" in state:
        extras.append(f"step={state['step']}")
    if "eval_score" in state:
        extras.append(f"score={state['eval_score']:.2f}")
    suffix = f" ({', '.join(extras)})" if extras else ""
    print(f"Loaded {ckpt_path.name} from {logdir}{suffix}")
    return agent, config
