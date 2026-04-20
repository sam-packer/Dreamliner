"""Evaluate a trained DreamerV3 run: learning curves + per-scenario recovery stats.

    uv run evaluate                              # latest run: plots + 200 eval rollouts per checkpoint (best and latest)
    uv run evaluate runs/dreamer/baseline        # specific run
    uv run evaluate --episodes 500               # bigger eval sample
    uv run evaluate --checkpoint best            # only best.pt
    uv run evaluate --checkpoint latest          # only latest.pt
    uv run evaluate --no-eval                    # plots only, skip rollouts

Both checkpoints face the same scenario draws (env is re-seeded per checkpoint)
so outcomes are directly comparable.

Outputs land under ``<logdir>/analysis/``:
    - learning_curves.png                learning curves (checkpoint-agnostic)
    - recovery_metrics_<ckpt>.png        altitude loss, length, success rate per scenario
    - eval_episodes_<ckpt>.json          raw per-episode trajectory log
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from dreamliner.envs import DreamerStallEnv
from dreamliner.evaluation._loader import find_latest_run, load_run, resolve_run_env_config
from dreamliner.evaluation.play import rollout_episodes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Learning curves and per-scenario recovery stats for a training run.",
    )
    p.add_argument("logdir", type=str, nargs="?", default=None,
                   help="runs/dreamer/<run> dir. Omit to auto-pick the latest run.")
    p.add_argument("--episodes", type=int, default=200,
                   help="Greedy eval episodes for recovery-metric histograms (default: 200).")
    p.add_argument("--checkpoint", choices=["best", "latest"], nargs="+",
                   default=["best", "latest"],
                   help="Which checkpoint(s) to roll out (default: both).")
    p.add_argument("--no-eval", action="store_true",
                   help="Skip rolling out the agent; only plot TensorBoard scalars.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TensorBoard scalar loading
# ---------------------------------------------------------------------------

def load_tb_scalars(logdir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(logdir), size_guidance={"scalars": 0})
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def plot_learning_curves(scalars: dict[str, list[tuple[int, float]]], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    interesting = [
        "episode/score", "episode/length",
        "episode/eval_score", "episode/eval_length",
        "train/loss/policy", "train/loss/value", "train/loss/world",
        "fps/fps",
    ]
    available = [t for t in interesting if t in scalars]
    if not available:
        # fall back to whatever is there
        available = sorted(scalars.keys())[:6]

    n = len(available)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
    for ax, tag in zip(axes.flat, available):
        steps = [s for s, _ in scalars[tag]]
        vals  = [v for _, v in scalars[tag]]
        ax.plot(steps, vals)
        ax.set_title(tag)
        ax.set_xlabel("env step")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(available):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-scenario recovery stats
# ---------------------------------------------------------------------------

def summarize_per_scenario(episodes: list[dict]) -> dict[str, dict]:
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_scenario[ep["scenario"]].append(ep)
    summary: dict[str, dict] = {}
    for name, eps in by_scenario.items():
        n = len(eps)
        success = sum(1 for e in eps if e["outcome"] == "success")
        crash   = sum(1 for e in eps if e["outcome"] == "crash")
        timeout = sum(1 for e in eps if e["outcome"] == "timeout")
        summary[name] = {
            "n": n,
            "success_rate": success / n if n else 0.0,
            "crash_rate":   crash   / n if n else 0.0,
            "timeout_rate": timeout / n if n else 0.0,
            "altitude_loss_ft": [e["altitude_loss_ft"] for e in eps],
            "length_steps":     [len(e["rewards"]) for e in eps],
            "return":           [e["total_reward"] for e in eps],
        }
    return summary


def plot_recovery_metrics(summary: dict[str, dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt
    scenarios = sorted(summary.keys())
    n = len(scenarios)
    fig, axes = plt.subplots(3, n, figsize=(3.0 * n, 9.0), squeeze=False)
    for col, name in enumerate(scenarios):
        s = summary[name]
        axes[0, col].hist(s["altitude_loss_ft"], bins=20, color="tab:red", alpha=0.7)
        axes[0, col].set_title(f"{name}\n(n={s['n']})")
        axes[0, col].set_xlabel("altitude lost (ft)")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].hist(s["length_steps"], bins=20, color="tab:blue", alpha=0.7)
        axes[1, col].set_xlabel("episode length (agent steps)")
        axes[1, col].grid(True, alpha=0.3)

        labels = ["success", "timeout", "crash"]
        values = [s["success_rate"], s["timeout_rate"], s["crash_rate"]]
        colors = ["tab:green", "tab:orange", "tab:red"]
        axes[2, col].bar(labels, values, color=colors)
        axes[2, col].set_ylim(0, 1)
        axes[2, col].set_xlabel("outcome rate")
        axes[2, col].grid(True, alpha=0.3, axis="y")

    axes[0, 0].set_ylabel("count")
    axes[1, 0].set_ylabel("count")
    axes[2, 0].set_ylabel("rate")
    fig.suptitle("Per-scenario recovery metrics", y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir) if args.logdir else find_latest_run()
    if not args.logdir:
        print(f"No logdir given; using latest run: {logdir}")
    out_dir = logdir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    env_config = resolve_run_env_config(logdir)

    print(f"Reading TensorBoard scalars from {logdir} ...")
    scalars = load_tb_scalars(logdir)
    print(f"  {len(scalars)} scalar tags found.")

    plot_learning_curves(scalars, out_dir / "learning_curves.png")
    print(f"  -> {out_dir / 'learning_curves.png'}")

    if args.no_eval:
        print("--no-eval: skipping rollout phase.")
        return

    # Bypass the training-time curriculum so every scenario gets roughly equal
    # sampling. Without this, the env would default to phase 0 (cruise only) and
    # the "per-scenario" summary would only report cruise.
    # Re-seeded per checkpoint below so both face the same scenario draws.
    env_seed = 12345
    for ckpt in args.checkpoint:
        print(f"\n=== Evaluating {ckpt}.pt ===")
        print(f"Loading agent from {logdir} ...")
        agent, config = load_run(logdir, prefer=ckpt)
        env = DreamerStallEnv(seed=env_seed, config=env_config, disable_curriculum=True)
        try:
            print(f"Running {args.episodes} greedy eval episodes from {ckpt}.pt ...")
            episodes = rollout_episodes(agent, env, args.episodes, config.device, progress=False)
        finally:
            env.close()

        episodes_path = out_dir / f"eval_episodes_{ckpt}.json"
        with open(episodes_path, "w", encoding="utf-8") as f:
            json.dump(episodes, f, indent=2)

        summary = summarize_per_scenario(episodes)
        print(f"\nPer-scenario summary ({ckpt}.pt):")
        for name in sorted(summary):
            s = summary[name]
            n = s["n"]
            if n == 0:
                continue
            med_alt = sorted(s["altitude_loss_ft"])[n // 2]
            print(
                f"  {name:20s}  n={n:3d}  success={s['success_rate']*100:5.1f}%  "
                f"crash={s['crash_rate']*100:5.1f}%  median_alt_loss={med_alt:6.0f}ft"
            )

        metrics_path = out_dir / f"recovery_metrics_{ckpt}.png"
        plot_recovery_metrics(summary, metrics_path)
        print(f"  -> {metrics_path}")
        print(f"  -> {episodes_path}")


if __name__ == "__main__":
    main()
