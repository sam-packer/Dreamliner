"""Roll out a trained DreamerV3 policy on StallRecoveryEnv.

    uv run play                                          # 5 episodes from latest run
    uv run play runs/dreamer/baseline                    # 5 episodes from a specific run
    uv run play --episodes 10                            # more episodes
    uv run play --flightgear                             # launch FlightGear + stream
    uv run play --flightgear --no-fg-launch              # you started FG; just stream
    uv run play --flightgear --fg-aircraft c172p         # different FG visual
    uv run play --out trajectories.json                  # save per-episode trajectory log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# vendor path setup
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "r2dreamer"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from tensordict import TensorDict  # noqa: E402

from dreamliner.envs import DreamerStallEnv  # noqa: E402
from dreamliner.evaluation._loader import find_latest_run, load_run, resolve_run_env_config  # noqa: E402
from dreamliner.utils.flightgear import launch_flightgear, wait_until_ready  # noqa: E402

_FG_READY_TIMEOUT_SECS = 180.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roll out a trained agent (greedy) on the stall-recovery env.",
    )
    p.add_argument("logdir", type=str, nargs="?", default=None,
                   help="runs/dreamer/<run> dir. Omit to auto-pick the latest run.")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of rollouts (default: 5).")
    p.add_argument("--flightgear", action="store_true",
                   help="Stream JSBSim state to FlightGear (UDP 5550, 60 Hz). "
                        "Auto-launches FG unless --no-fg-launch is set.")
    p.add_argument("--no-fg-launch", action="store_true",
                   help="With --flightgear: skip auto-launch (you started FG yourself).")
    p.add_argument("--fg-aircraft", type=str, default="737-300",
                   help="FG visual model (default: 737-300). Must be installed "
                        "via the FG launcher first - see README.")
    p.add_argument("--out", type=str, default=None,
                   help="Write per-episode trajectory log to this JSON path.")
    return p.parse_args()


def _obs_to_trans(obs: dict, device: str) -> TensorDict:
    """Match the (B=1, *) shape that r2dreamer's agent.act expects."""
    tensors = {k: torch.as_tensor(np.asarray(v)).unsqueeze(0) for k, v in obs.items()}
    td = TensorDict(tensors, batch_size=(1,))
    # lift_dim equivalent: any 1-D entry gets a trailing axis.
    for key in td.keys():
        if td[key].ndim == 1:
            td[key] = td[key].unsqueeze(-1)
    return td.to(device, non_blocking=True)


def rollout_episodes(
    agent,
    env: DreamerStallEnv,
    num_episodes: int,
    device: str,
    progress: bool = True,
) -> list[dict]:
    """Run ``num_episodes`` greedy rollouts; return per-episode trajectory dicts."""
    episodes: list[dict] = []
    for ep in range(num_episodes):
        obs = env.reset()
        agent_state = agent.get_initial_state(1)
        log = {
            "episode": ep,
            "scenario": None,
            "total_reward": 0.0,
            "altitude_loss_ft": None,
            "outcome": None,
            "alpha":    [],
            "altitude": [],
            "vc":       [],
            "roll":     [],
            "rewards":  [],
        }

        info: dict = {}
        done = False
        while not done:
            trans = _obs_to_trans(obs, device)
            with torch.no_grad():
                # Third positional arg is `eval=True` (greedy mode action).
                action, agent_state = agent.act(trans, agent_state, True)
            action_np = action[0].detach().cpu().numpy().astype(np.float32)
            obs, reward, done, info = env.step(action_np)
            log["alpha"].append(float(info["alpha_deg"]))
            log["altitude"].append(float(info["altitude_ft"]))
            log["vc"].append(float(info["vc_kts"]))
            log["roll"].append(float(info["roll_deg"]))
            log["rewards"].append(float(reward))
            log["total_reward"] += float(reward)

        log["scenario"] = info["scenario"]
        log["altitude_loss_ft"] = float(info["altitude_loss_ft"])
        log["outcome"] = "crash" if info["crashed"] else ("success" if info["success"] else "timeout")
        if progress:
            print(
                f"ep {ep:2d}  {log['scenario']:20s}  return={log['total_reward']:8.2f}  "
                f"alt_loss={log['altitude_loss_ft']:6.0f}ft  steps={len(log['rewards']):3d}  "
                f"outcome={log['outcome']}"
            )
        episodes.append(log)
    return episodes


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir) if args.logdir else find_latest_run()
    if not args.logdir:
        print(f"No logdir given; using latest run: {logdir}")
    agent, config = load_run(logdir)
    device = config.device
    env_config = resolve_run_env_config(logdir)

    if args.flightgear and not args.no_fg_launch:
        proc = launch_flightgear(aircraft=args.fg_aircraft)
        print(f"Launched FlightGear (PID {proc.pid}, visual={args.fg_aircraft}); "
              f"waiting for scenery load (up to {_FG_READY_TIMEOUT_SECS:.0f}s)...")
        elapsed = wait_until_ready(timeout=_FG_READY_TIMEOUT_SECS)
        print(f"FlightGear ready after {elapsed:.1f}s; starting agent rollouts...")
    elif args.flightgear:
        print("FlightGear streaming -> UDP localhost:5550. Waiting for FG to be ready...")
        elapsed = wait_until_ready(timeout=_FG_READY_TIMEOUT_SECS)
        print(f"FlightGear ready after {elapsed:.1f}s; starting agent rollouts...")

    env = DreamerStallEnv(seed=0, config=env_config, flightgear=args.flightgear)
    try:
        episodes = rollout_episodes(agent, env, args.episodes, device)
    finally:
        env.close()

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2)
        print(f"\nWrote trajectory log to {out_path}")


if __name__ == "__main__":
    main()
