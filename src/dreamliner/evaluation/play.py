"""Roll out a trained DreamerV3 policy on StallRecoveryEnv.

    uv run play                                          # 5 episodes from latest run
    uv run play runs/dreamer/baseline                    # 5 episodes from a specific run
    uv run play --episodes 10                            # more episodes
    uv run play --demo                                   # each configured scenario once, in order
    uv run play --flightgear                             # launch FlightGear + stream in real time
    uv run play --flightgear --demo                      # cockpit/chase replay for each scenario once
    uv run play --flightgear --no-fg-launch              # you started FG; just stream
    uv run play --flightgear --fg-aircraft c172p         # different FG visual
    uv run play --flightgear --episodes 1 --scenario turning_stall
    uv run play --out trajectories.json                  # save per-episode trajectory log
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
from dreamliner.utils.flightgear import configure_inspection_view, launch_flightgear, wait_until_ready  # noqa: E402

_FG_READY_TIMEOUT_SECS = 180.0
_DEFAULT_FG_REPLAY_VIEWS = ("cockpit", "chase")
_DEFAULT_FG_COCKPIT_FOV = 90.0
_SUCCESS_NARRATION = "Success"
_DEMO_EXCLUDED_SCENARIOS = frozenset({
    "cruise",
    "gentle_turn",
    "pitch_recovery",
    "slow_flight",
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Roll out a trained agent (greedy) on the stall-recovery env.",
    )
    p.add_argument("logdir", type=str, nargs="?", default=None,
                   help="runs/dreamer/<run> dir. Omit to auto-pick the latest run.")
    p.add_argument("--episodes", type=int, default=5,
                   help="Number of rollouts (default: 5).")
    p.add_argument("--demo", action="store_true",
                   help="Run each configured scenario exactly once, in config order.")
    p.add_argument("--flightgear", action="store_true",
                   help="Stream JSBSim state to FlightGear (UDP 5550, 60 Hz). "
                        "Auto-launches FG unless --no-fg-launch is set.")
    p.add_argument("--no-fg-launch", action="store_true",
                   help="With --flightgear: skip auto-launch (you started FG yourself).")
    p.add_argument("--fg-aircraft", type=str, default="737-300",
                   help="FG visual model (default: 737-300). Must be installed "
                        "via the FG launcher first - see README.")
    p.add_argument("--fg-replay-views", type=str,
                   default=",".join(_DEFAULT_FG_REPLAY_VIEWS),
                   help="With --flightgear: comma-separated replay view sequence "
                        "(default: cockpit,chase).")
    p.add_argument("--fg-cockpit-fov", type=float, default=_DEFAULT_FG_COCKPIT_FOV,
                   help="With --flightgear: cockpit field-of-view in degrees for inspection "
                        "replays (default: 90). Larger = more zoomed out.")
    p.add_argument("--scenario", type=str, default=None,
                   help="Force every episode to start in this scenario name from env_config.yaml.")
    p.add_argument("--out", type=str, default=None,
                   help="Write per-episode trajectory log to this JSON path.")
    return p.parse_args()


def _resolve_curriculum_step_file(logdir: Path) -> Path | None:
    candidate = logdir / "_curriculum_step.txt"
    return candidate if candidate.exists() else None


def _parse_view_sequence(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _demo_scenario_names(scenario_names: tuple[str, ...]) -> list[str]:
    names = [name for name in scenario_names if name not in _DEMO_EXCLUDED_SCENARIOS]
    return names or list(scenario_names)


def _speak_async(text: str) -> None:
    if sys.platform != "win32":
        return
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        return
    escaped_text = text.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Speak('{escaped_text}')"
    )
    try:
        subprocess.Popen(
            [powershell, "-NoProfile", "-NonInteractive", "-Command", command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        pass


def _scenario_narration(scenario: str | None) -> str | None:
    if not scenario:
        return None
    return scenario.replace("_", " ")


def _maybe_announce_scenario(
    scenario: str | None,
    *,
    narration_state: dict[str, str | None] | None,
) -> None:
    narration = _scenario_narration(scenario)
    if narration is None:
        return
    if narration_state is not None and narration_state.get("last_scenario") == scenario:
        return
    _speak_async(narration)
    if narration_state is not None:
        narration_state["last_scenario"] = scenario


def _print_episode_start(
    ep: int,
    reset_info: dict[str, Any],
    *,
    max_episode_seconds: float,
    success_hold_seconds: float,
    header: str | None = None,
) -> None:
    scenario = str(reset_info.get("scenario", "?"))
    curriculum_step = int(reset_info.get("curriculum_step", 0))
    ics = dict(reset_info.get("ics") or {})
    tag = f"\nep {ep:2d}"
    if header:
        tag += f"  {header}"
    print(
        f"{tag} start  scenario={scenario:20s}  curriculum_step={curriculum_step}  "
        f"max_time={max_episode_seconds:4.1f}s  success_hold={success_hold_seconds:3.1f}s"
    )
    if not ics:
        return
    print(
        "        "
        f"init alt={ics['altitude_ft']:7.0f}ft  vc={ics['airspeed_kcas']:6.1f}kt  "
        f"alpha={ics['alpha_deg']:5.1f}deg  pitch={ics['pitch_deg']:6.1f}deg  "
        f"roll={ics['roll_deg']:6.1f}deg"
    )
    print(
        "        "
        f"init beta={ics['beta_deg']:5.1f}deg  yaw_rate={ics['yaw_rate_dps']:6.1f}deg/s  "
        f"throttle={ics['throttle']:.2f}"
    )


def _print_episode_status(
    info: dict[str, Any],
    *,
    sim_time_seconds: float,
    stable_seconds: float,
    success_hold_seconds: float,
) -> None:
    stalled = "yes" if info["stalled"] else "no"
    print(
        "        "
        f"t={sim_time_seconds:5.1f}s  alpha={info['alpha_deg']:5.1f}deg  vc={info['vc_kts']:6.1f}kt  "
        f"roll={info['roll_deg']:6.1f}deg  alt={info['altitude_ft']:7.0f}ft  "
        f"alt_loss={info['altitude_loss_ft']:6.0f}ft  stalled={stalled:3s}  "
        f"stable={stable_seconds:3.1f}/{success_hold_seconds:3.1f}s"
    )


def _rollout_single_episode(
    agent,
    env: DreamerStallEnv,
    device: str,
    *,
    episode_index: int,
    progress: bool,
    scenario: str | None,
    initial_conditions: dict[str, float] | None,
    status_interval_steps: int | None,
    header: str | None = None,
    replay_index: int | None = None,
    replay_count: int | None = None,
    replay_view: str | None = None,
    replay_view_name: str | None = None,
    announce_scenario: bool = False,
    narration_state: dict[str, str | None] | None = None,
) -> dict:
    agent_dt_hz = env.agent_dt_hz
    success_hold_seconds = env.success_hold_seconds
    max_episode_seconds = env.max_episode_seconds
    reset_options: dict[str, Any] = {}
    if scenario is not None:
        reset_options["scenario"] = scenario
    if initial_conditions is not None:
        reset_options["initial_conditions"] = initial_conditions
    obs = env.reset(options=reset_options or None)
    agent_state = agent.get_initial_state(1)
    reset_info = dict(env.last_reset_info)
    ics = dict(reset_info.get("ics") or {})
    log = {
        "episode": episode_index,
        "scenario": reset_info.get("scenario"),
        "curriculum_step": int(reset_info.get("curriculum_step", 0)),
        "initial_conditions": ics,
        "total_reward": 0.0,
        "altitude_loss_ft": None,
        "outcome": None,
        "sim_time_seconds": None,
        "success_hold_seconds": success_hold_seconds,
        "max_episode_seconds": max_episode_seconds,
        "stable_streak_seconds": 0.0,
        "final_state": {},
        "replay_index": replay_index,
        "replay_count": replay_count,
        "replay_view": replay_view,
        "replay_view_name": replay_view_name,
        "alpha": [],
        "altitude": [],
        "vc": [],
        "roll": [],
        "rewards": [],
    }
    if progress:
        _print_episode_start(
            episode_index,
            reset_info,
            max_episode_seconds=max_episode_seconds,
            success_hold_seconds=success_hold_seconds,
            header=header,
        )
    if announce_scenario:
        _maybe_announce_scenario(str(log["scenario"]) if log["scenario"] is not None else None,
                                 narration_state=narration_state)

    info: dict = {}
    done = False
    next_status_step = status_interval_steps
    while not done:
        trans = _obs_to_trans(obs, device)
        with torch.no_grad():
            action, agent_state = agent.act(trans, agent_state, True)
        action_np = action[0].detach().cpu().numpy().astype(np.float32)
        obs, reward, done, info = env.step(action_np)
        log["alpha"].append(float(info["alpha_deg"]))
        log["altitude"].append(float(info["altitude_ft"]))
        log["vc"].append(float(info["vc_kts"]))
        log["roll"].append(float(info["roll_deg"]))
        log["rewards"].append(float(reward))
        log["total_reward"] += float(reward)
        if progress and next_status_step is not None and len(log["rewards"]) >= next_status_step:
            sim_time_seconds = len(log["rewards"]) / agent_dt_hz
            stable_seconds = float(info["stable_streak_steps"]) / agent_dt_hz
            _print_episode_status(
                info,
                sim_time_seconds=sim_time_seconds,
                stable_seconds=stable_seconds,
                success_hold_seconds=success_hold_seconds,
            )
            next_status_step += status_interval_steps

    log["scenario"] = info["scenario"]
    log["altitude_loss_ft"] = float(info["altitude_loss_ft"])
    log["outcome"] = "crash" if info["crashed"] else ("success" if info["success"] else "timeout")
    log["sim_time_seconds"] = len(log["rewards"]) / agent_dt_hz
    log["stable_streak_seconds"] = float(info["stable_streak_steps"]) / agent_dt_hz
    log["final_state"] = {
        "alpha_deg": float(info["alpha_deg"]),
        "altitude_ft": float(info["altitude_ft"]),
        "altitude_loss_ft": float(info["altitude_loss_ft"]),
        "roll_deg": float(info["roll_deg"]),
        "vc_kts": float(info["vc_kts"]),
        "stalled": bool(info["stalled"]),
    }
    if progress:
        done_tag = f"ep {episode_index:2d}"
        if header:
            done_tag += f"  {header}"
        print(
            f"{done_tag} done   {log['scenario']:20s}  return={log['total_reward']:8.2f}  "
            f"alt_loss={log['altitude_loss_ft']:6.0f}ft  steps={len(log['rewards']):3d}  "
            f"sim_time={log['sim_time_seconds']:5.1f}s  outcome={log['outcome']}"
        )
        print(
            "        "
            f"final alpha={info['alpha_deg']:5.1f}deg  vc={info['vc_kts']:6.1f}kt  "
            f"roll={info['roll_deg']:6.1f}deg  alt={info['altitude_ft']:7.0f}ft  "
            f"stable={log['stable_streak_seconds']:3.1f}/{success_hold_seconds:3.1f}s"
        )
    return log


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
    *,
    announce_success: bool = False,
    announce_scenario: bool = False,
    scenario: str | None = None,
    status_interval_steps: int | None = None,
) -> list[dict]:
    """Run ``num_episodes`` greedy rollouts; return per-episode trajectory dicts."""
    episodes: list[dict] = []
    narration_state = {"last_scenario": None} if announce_scenario else None
    for ep in range(num_episodes):
        log = _rollout_single_episode(
            agent,
            env,
            device,
            episode_index=ep,
            progress=progress,
            scenario=scenario,
            initial_conditions=None,
            status_interval_steps=status_interval_steps,
            announce_scenario=announce_scenario,
            narration_state=narration_state,
        )
        episodes.append(log)
        if announce_success and log["outcome"] == "success":
            _speak_async(_SUCCESS_NARRATION)
    return episodes


def rollout_scenario_sequence(
    agent,
    env: DreamerStallEnv,
    scenario_names: list[str],
    device: str,
    progress: bool = True,
    *,
    announce_success: bool = False,
    announce_scenario: bool = False,
    status_interval_steps: int | None = None,
) -> list[dict]:
    episodes: list[dict] = []
    narration_state = {"last_scenario": None} if announce_scenario else None
    for ep, scenario_name in enumerate(scenario_names):
        log = _rollout_single_episode(
            agent,
            env,
            device,
            episode_index=ep,
            progress=progress,
            scenario=scenario_name,
            initial_conditions=None,
            status_interval_steps=status_interval_steps,
            announce_scenario=announce_scenario,
            narration_state=narration_state,
        )
        episodes.append(log)
        if announce_success and log["outcome"] == "success":
            _speak_async(_SUCCESS_NARRATION)
    return episodes


def rollout_flightgear_replays(
    agent,
    env: DreamerStallEnv,
    num_episodes: int,
    device: str,
    *,
    announce_success: bool = False,
    replay_views: list[str],
    cockpit_fov: float,
    scenario: str | None = None,
    progress: bool = True,
    status_interval_steps: int | None = None,
) -> list[dict]:
    episodes: list[dict] = []
    replay_count = len(replay_views)
    narration_state = {"last_scenario": None}
    for ep in range(num_episodes):
        replay_scenario = scenario
        replay_ics: dict[str, float] | None = None
        last_log: dict[str, Any] | None = None
        for replay_index, requested_view in enumerate(replay_views):
            actual_view = configure_inspection_view(requested_view, cockpit_fov=cockpit_fov)
            header = f"replay={replay_index + 1}/{replay_count}  view={requested_view}->{actual_view}"
            log = _rollout_single_episode(
                agent,
                env,
                device,
                episode_index=ep,
                progress=progress,
                scenario=replay_scenario,
                initial_conditions=replay_ics,
                status_interval_steps=status_interval_steps,
                header=header,
                replay_index=replay_index,
                replay_count=replay_count,
                replay_view=requested_view,
                replay_view_name=actual_view,
                announce_scenario=(replay_index == 0),
                narration_state=narration_state,
            )
            if replay_index == 0:
                replay_scenario = str(log["scenario"])
                replay_ics = dict(log["initial_conditions"])
            episodes.append(log)
            last_log = log
            time.sleep(0.25)
        if announce_success and last_log is not None and last_log["outcome"] == "success":
            _speak_async(_SUCCESS_NARRATION)
    return episodes


def rollout_flightgear_demo_replays(
    agent,
    env: DreamerStallEnv,
    scenario_names: list[str],
    device: str,
    *,
    announce_success: bool = False,
    replay_views: list[str],
    cockpit_fov: float,
    progress: bool = True,
    status_interval_steps: int | None = None,
) -> list[dict]:
    episodes: list[dict] = []
    replay_count = len(replay_views)
    narration_state = {"last_scenario": None}
    for ep, scenario_name in enumerate(scenario_names):
        last_log: dict[str, Any] | None = None
        replay_ics: dict[str, float] | None = None
        for replay_index, requested_view in enumerate(replay_views):
            actual_view = configure_inspection_view(requested_view, cockpit_fov=cockpit_fov)
            header = f"replay={replay_index + 1}/{replay_count}  view={requested_view}->{actual_view}"
            log = _rollout_single_episode(
                agent,
                env,
                device,
                episode_index=ep,
                progress=progress,
                scenario=scenario_name,
                initial_conditions=replay_ics,
                status_interval_steps=status_interval_steps,
                header=header,
                replay_index=replay_index,
                replay_count=replay_count,
                replay_view=requested_view,
                replay_view_name=actual_view,
                announce_scenario=(replay_index == 0),
                narration_state=narration_state,
            )
            if replay_index == 0:
                replay_ics = dict(log["initial_conditions"])
            episodes.append(log)
            last_log = log
            time.sleep(0.25)
        if announce_success and last_log is not None and last_log["outcome"] == "success":
            _speak_async(_SUCCESS_NARRATION)
    return episodes


def main() -> None:
    args = parse_args()
    if args.demo and args.scenario is not None:
        raise ValueError("--demo cannot be combined with --scenario")
    logdir = Path(args.logdir) if args.logdir else find_latest_run()
    if not args.logdir:
        print(f"No logdir given; using latest run: {logdir}")
    try:
        agent, config = load_run(logdir, prefer="last_good")
    except FileNotFoundError:
        print(f"No last_good.pt in {logdir}; falling back to best/latest.")
        agent, config = load_run(logdir, prefer="best")
    device = config.device
    env_config = resolve_run_env_config(logdir)
    curriculum_step_file = _resolve_curriculum_step_file(logdir)
    if curriculum_step_file is not None:
        print(f"Using curriculum step snapshot: {curriculum_step_file}")
    else:
        print(f"No curriculum step snapshot found in {logdir}; curriculum-enabled runs will sample phase 0.")

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

    replay_views = _parse_view_sequence(args.fg_replay_views)
    if args.flightgear:
        print(f"FlightGear inspection replay views: {', '.join(replay_views)}")
        print(f"Cockpit inspection FOV: {args.fg_cockpit_fov:.1f} deg")

    env = DreamerStallEnv(
        seed=0,
        config=env_config,
        flightgear=args.flightgear,
        curriculum_step_file=curriculum_step_file,
    )
    try:
        if args.demo:
            scenario_names = _demo_scenario_names(env.scenario_names)
            print(f"Demo mode: {len(scenario_names)} stall/upset scenarios in config order.")
            if args.flightgear and replay_views:
                episodes = rollout_flightgear_demo_replays(
                    agent,
                    env,
                    scenario_names,
                    device,
                    announce_success=True,
                    replay_views=replay_views,
                    cockpit_fov=args.fg_cockpit_fov,
                    status_interval_steps=env.agent_dt_hz,
                )
            else:
                episodes = rollout_scenario_sequence(
                    agent,
                    env,
                    scenario_names,
                    device,
                    announce_success=args.flightgear,
                    announce_scenario=args.flightgear,
                    status_interval_steps=env.agent_dt_hz if args.flightgear else None,
                )
        elif args.flightgear and replay_views:
            episodes = rollout_flightgear_replays(
                agent,
                env,
                args.episodes,
                device,
                announce_success=True,
                replay_views=replay_views,
                cockpit_fov=args.fg_cockpit_fov,
                scenario=args.scenario,
                status_interval_steps=env.agent_dt_hz,
            )
        else:
            episodes = rollout_episodes(
                agent,
                env,
                args.episodes,
                device,
                announce_success=args.flightgear,
                announce_scenario=args.flightgear,
                scenario=args.scenario,
                status_interval_steps=env.agent_dt_hz if args.flightgear else None,
            )
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
