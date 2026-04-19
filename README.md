# Dreamliner

Reinforcement learning for **commercial-aircraft stall recovery** using a **world-model** approach (DreamerV3) on top of the JSBSim flight dynamics engine. The novelty: every published RL stall/upset paper uses model-free methods (PPO/SAC). DreamerV3 learning a latent dynamics model of post-stall aerodynamics is an open research question.

## Why

Loss of control in flight (LOC-I) is the leading cause of fatal commercial aviation accidents (Air France 447 being the canonical example). The FAA now requires stall-recovery training in simulators. The interesting research question here is whether a world-model agent can learn the nonlinear attached-to-separated-flow transition and discover recovery strategies that inform autopilot design or a real-time pilot advisory system.

## Architecture

```
JSBSim (flight dynamics, in-process Python bindings, ~1000x real-time)
    +
StallRecoveryEnv (Gymnasium env, this repo)
    +
DreamerStallEnv (adapter: gym v0.21 + is_first/is_last/is_terminal flags)
    +
r2dreamer (vendored PyTorch DreamerV3 reproduction by NM512)
    +
RTX 5090 / CUDA 13 / torch 2.11+cu130
```

JSBSim runs **in-process** through its Python bindings — no network round-trip during training. The vendored DreamerV3 implementation is NM512's [r2dreamer](https://github.com/NM512/r2dreamer) (newer + faster than the older `dreamerv3-torch`); the official danijar/dreamerv3 is JAX-only and JAX has no Windows GPU wheels, so PyTorch is the path on this stack.

## Setup

Requires Python 3.12, NVIDIA GPU + CUDA 13 (RTX 5090 here), and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

`uv sync` installs everything including a CUDA 13 build of PyTorch (`torch==2.11.0+cu130`) via the configured `pytorch-cu130` index. The `r2dreamer` source lives under `vendor/r2dreamer/` (cloned from NM512's repo at first project setup) and is added to `sys.path` at runtime rather than installed — none of its env-suite-specific deps (mujoco, dm_control, atari, crafter, etc.) are needed.

## Commands

Four commands. Each defaults to "do the obvious thing" and auto-discovers the most recent run where relevant.

### Train

```bash
uv run train                                           # quick profile (~2.7 hr)
uv run train --profile good                            # good profile (~16 hr)
uv run train --run-name baseline                       # custom output dir name
uv run train --resume-from runs/dreamer/prior --run-name continued
```

Two profiles (RTX 5090 timings; both use `action_repeat=2`):

| profile | steps | model | wall-clock | VRAM      |
|---------|-------|-------|------------|-----------|
| `quick` | 250 k | 100 M | ~2.7 hr    | 22–25 GB  |
| `good`  | 1 M   | 200 M | ~20 hr     | near-max  |

`action_repeat=2` means the agent decides every 200 ms of sim time (vs 100 ms without it) — halves the required agent decisions to cover the same env duration. Matches the DreamerV3 norm for proprio / DMC-style tasks and stays well inside a human reaction-time envelope for stall recovery.

Tuning knobs (batch size, train ratio, buffer size, eval cadence, etc.) live in the `PROFILES` dict in `src/dreamliner/training/train.py` — edit there if you need something off-menu.

Outputs land under `runs/dreamer/<run-name>/`. `latest.pt` is saved every 10 k env steps, on clean exit, **and on Ctrl+C** — interrupts don't lose the model. `best.pt` is saved whenever eval score improves, with a `best.json` sidecar (step, score, timestamp). `play` and `evaluate` load `best.pt` by default and fall back to `latest.pt`.

`--resume-from` loads weights only; optimizer state and replay buffer are rebuilt fresh. This is simpler than snapshotting the ~12 GB GPU replay buffer and works fine for the "kill a run, extend it later" iteration loop.

### Play

Roll out a trained policy (greedy) against random stall scenarios.

```bash
uv run play                                            # 5 episodes, latest run, no FG
uv run play runs/dreamer/baseline                      # specific run
uv run play --episodes 10                              # more rollouts
uv run play --flightgear                               # with FlightGear (auto-launch)
uv run play --flightgear --no-fg-launch                # FG already running; just stream
uv run play --out trajectories.json                    # save per-episode log
```

Per-episode line goes to stdout: scenario, return, altitude lost, step count, outcome (success / timeout / crash).

### Evaluate

Learning curves + per-scenario recovery stats over many rollouts.

```bash
uv run evaluate                                        # latest run: plots + 50 rollouts
uv run evaluate runs/dreamer/baseline                  # specific run
uv run evaluate --episodes 200                         # bigger eval sample
uv run evaluate --no-eval                              # plots only, skip rollouts
```

Outputs land under `<logdir>/analysis/`:

- `learning_curves.png` — eval/train scalars from TensorBoard
- `recovery_metrics.png` — altitude-loss / length / success-rate histograms per scenario
- `eval_episodes.json` — raw per-episode trajectory log

### FlightGear

Standalone FG launcher. Use when you want the cockpit up before running `play --flightgear --no-fg-launch`, or just to verify the visual stack.

```bash
uv run flightgear                                      # launch FG with 737-300
uv run flightgear --aircraft c172p                     # Cessna 172P visual instead
```

JSBSim drives the actual flight dynamics; the FG aircraft is only what you see on screen. Chase-cam (`v` in FG) is the clearest "watch the recovery maneuver" view; cockpit (`ctrl+v` to cycle) gives you the attitude indicator + airspeed tape.

## FlightGear setup (one-time, ~5 min)

Install [FlightGear](https://www.flightgear.org/download/) separately (not a pip dependency). We auto-discover `fgfs` across the standard install locations (`C:\Program Files\FlightGear*\bin\fgfs.exe`, `/usr/games/fgfs`, `/Applications/FlightGear.app/...`) — no need to add it to PATH.

The default FG install only ships with the Cessna 172P; the 737-300 visual has to be downloaded once:

1. Open the FlightGear launcher (run `fgfs.exe` with no arguments, or use the Start menu shortcut).
2. Click the **Aircraft** tab on the left sidebar.
3. Click **Add-on Hangars** at the bottom of the aircraft list.
4. Make sure the **FlightGear 2024.1 Aircraft** hangar is checked (it should already be, since FG 2024 ships with it pre-configured pointing at `https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/catalog.xml`). Click **Update All** to refresh the catalog.
5. Back in the Aircraft list, type `737-300` in the search box at the top.
6. Click the 737-300 entry, then **Install** in the right pane. Wait for the download (~50 MB).
7. Once installed, close the launcher.

If FG 2024.1's catalog isn't pre-configured for some reason: in step 4, click **Add hangar** and paste `https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/catalog.xml`, then continue with step 5.

## Environment

`StallRecoveryEnv` randomly initialises the 737 in one of five stall scenarios per episode and asks the policy to recover with minimum altitude loss.

**Scenarios** (defined in `configs/default.yaml`):

| name                | description                                                |
|---------------------|------------------------------------------------------------|
| `high_alpha_entry`  | nose-up attitude, decaying airspeed, approaching stall AoA |
| `wings_level_stall` | classic low-speed wings-level stall                        |
| `turning_stall`     | banked-turn stall (20–60 deg bank)                         |
| `nose_high_upset`   | extreme pitch (30–60 deg) with low airspeed                |
| `incipient_spin`    | high AoA with significant yaw rate                         |

Initial conditions (altitude 5–25 kft, airspeed, alpha, pitch, roll, beta, yaw rate, throttle) are sampled per-episode from per-scenario ranges in the YAML.

**Observation** (Dict, with a single `state` key for `mlp_keys=state` on Dreamer):

```text
state       : np.ndarray, shape (15,), dtype float32
is_first    : bool
is_last     : bool
is_terminal : bool
```

The 15-dim normalized state vector: alpha, beta, pitch, roll, calibrated airspeed, altitude, body angular rates (p, q, r), normal load factor, throttle, elevator/aileron/rudder positions, vertical speed.

**Action** (`Box(-1, 1, shape=(4,))`): elevator, aileron, rudder, throttle. Throttle is internally remapped from [-1, 1] to [0, 1] before being sent to JSBSim.

**Reward** (per agent step):

- penalty for alpha above the safe threshold
- penalty for altitude lost from episode start
- penalty for roll away from wings-level
- bonus for airspeed above stall speed
- penalty for jerky control inputs (smoothness regularizer)
- one-shot **-100** on crash (alt < 1000 ft and descending)
- one-shot **+50** when stable flight is held for 5 s

All weights live in `configs/default.yaml` under `reward:` and `targets:`.

**Termination**: crash, sustained recovery (5 s), or 60 s timeout.

**Timing**: JSBSim integrates at 120 Hz; the agent acts at 10 Hz, so each `env.step()` runs 12 JSBSim substeps.

## Configuration

`src/dreamliner/configs/default.yaml` controls scenarios, reward weights, recovery targets, and observation normalization scales. The env loads this file by default; pass a different file or a Python dict to `StallRecoveryEnv(config=...)` to override.

DreamerV3 model + trainer hyperparameters are pulled from `vendor/r2dreamer/configs/model/_base_.yaml` and the per-profile `size{12M,25M,...,400M}.yaml` file, then overridden in `train.py:build_config` with our env-side knobs.

## Project layout

```
src/dreamliner/
  configs/default.yaml          # env scenarios + reward weights
  data/flightgear.xml           # JSBSim output directive (UDP 5550, 60 Hz)
  envs/
    stall_recovery_env.py       # core Gymnasium env (737 + JSBSim)
    dreamer_adapter.py          # gym v0.21 + is_first/is_last/is_terminal shim
  training/
    train.py                    # uv run train
  evaluation/
    _loader.py                  # rebuild Dreamer agent from a run dir
    play.py                     # uv run play
    evaluate.py                 # uv run evaluate
  utils/
    flightgear.py               # uv run flightgear + FG discovery
    jsbsim_utils.py             # FDM construction, scenario sampling, XML patching
vendor/r2dreamer/               # cloned PyTorch DreamerV3 reproduction
pyproject.toml                  # deps, scripts, CUDA 13 torch index
```

## Notes and gotchas

- **JSBSim XML input sockets**: the bundled `737.xml` declares two listening input sockets (telnet 5137, QTJSBSIM 5139). With multiple parallel envs — or in any process where one already binds them — subsequent FDMs spam `Could not bind to TCP/UDP input socket`. We copy the aircraft dir to a per-process temp location and strip those declarations once. See `dreamliner.utils.jsbsim_utils:_patched_aircraft_path`.
- **737 model approximation**: JSBSim's bundled 737 is built from public data, not Boeing proprietary. Stall behaviour is approximate but adequate for research. Swap to `c172p` in the YAML if you want better-validated stall characteristics.
- **Aileron position**: the 737 model exposes `fcs/left-aileron-pos-norm` and `fcs/right-aileron-pos-norm` but not a combined `fcs/aileron-pos-norm`. The env averages them.
- **Engine startup**: every reset calls `fdm.get_propulsion().init_running(-1)` so thrust is available immediately rather than spooling up over several seconds.
- **JSBSim reset**: in-place reset across very different ICs is unreliable, so the env rebuilds `FGFDMExec` on every `reset()`. Construction is cheap (sub-second).
- **JAX on Windows**: not an option for GPU. JAX has no Windows GPU wheels (cu12 or cu13). The official `danijar/dreamerv3` would require WSL2; we use NM512's PyTorch reproduction instead.
- **`uv run` rebuild lock on Windows**: `uv run train` and `uv add` rebuild the project wheel, which tries to replace `train.exe` in the venv. If another `train` is still running, that file is locked and the invocation errors with `The process cannot access the file because it is being used by another process`. Fix: Ctrl+C the running training (the next `latest.pt` has already been saved), then re-run.

## What is coming next

1. Latent-space visualization (t-SNE / UMAP of RSSM stoch+deter colored by stall phase) — the "what did the world model learn" research figure.
2. Reward / scenario tuning based on the per-scenario recovery metrics from `evaluate`.
