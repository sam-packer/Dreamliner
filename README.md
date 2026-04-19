# Dreamliner

Reinforcement learning for **commercial-aircraft stall recovery** using a **world-model** approach (DreamerV3) on top of
the JSBSim flight dynamics engine. The novelty: every published RL stall/upset paper uses model-free methods (PPO/SAC).
DreamerV3 learning a latent dynamics model of post-stall aerodynamics is an open research question.

## Why

Loss of control in flight (LOC-I) is the leading cause of fatal commercial aviation accidents (Air France 447 being the
canonical example). The FAA now requires stall-recovery training in simulators. The interesting research question here
is whether a world-model agent can learn the nonlinear attached-to-separated-flow transition and discover recovery
strategies that inform autopilot design or a real-time pilot advisory system.

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

JSBSim runs **in-process** through its Python bindings (no network round-trip during training). The vendored DreamerV3
implementation is NM512's [r2dreamer](https://github.com/NM512/r2dreamer) (newer and faster than the older
`dreamerv3-torch`); the official danijar/dreamerv3 is JAX-only and JAX has no Windows GPU wheels, so PyTorch is the path
on this stack.

## Setup

Requires Python 3.12, NVIDIA GPU + CUDA 13 (RTX 5090 here), and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.get_device_name(0))"
```

`uv sync` installs everything including a CUDA 13 build of PyTorch (`torch==2.11.0+cu130`) via the configured
`pytorch-cu130` index, plus `triton-windows` so `torch.compile` works on Windows. The `r2dreamer` source lives under
`vendor/r2dreamer/` (cloned from NM512's repo at first project setup) and is added to `sys.path` at runtime rather than
installed. None of its env-suite-specific deps (mujoco, dm_control, atari, crafter, etc.) are needed.

For full `torch.compile` coverage on Windows, install **Visual Studio 2022 Build Tools** with the "Desktop development
with C++" workload. `train.py` auto-detects it and sources `vcvars64.bat` at startup so `cl.exe` lands on PATH for
Inductor's C++ codegen. Without Build Tools, compile still works via the Triton-only path but some ops fall back to
eager mode and you lose part of the speedup.

## Commands

Four commands. Each defaults to "do the obvious thing" and auto-discovers the most recent run where relevant.

### Train

```bash
uv run train                                           # quick profile (~50 min)
uv run train --profile good                            # good profile (~4 hr)
uv run train --run-name baseline                       # custom output dir name
uv run train --resume-from runs/dreamer/prior --run-name continued
```

Two profiles (RTX 5090 + Ryzen 9 9950X3D timings):

| profile | steps | model | envs | wall-clock | VRAM     |
|---------|-------|-------|------|------------|----------|
| `quick` | 250 k | 100 M | 16   | ~50 min    | 22–25 GB |
| `good`  | 1 M   | 100 M | 32   | ~3.5 hrs   | 22–25 GB |

`good` is `quick` extended: same model, same VRAM load, 4× more environment steps and 2× the parallel env workers to
saturate both GPU compute and CPU physics threads. For state-based tasks at this scale (15-dim obs, 4-dim action), more
data dominates more model capacity — a 200M model variant exists but didn't justify its 2× compute cost here.

`action_repeat=2` appears in the trainer config but is **display-only** in this codebase — our custom env isn't wrapped
with an action-repeat wrapper (only the DMC/Atari/metaworld envs in `vendor/r2dreamer/envs/` are). The policy decides
every physical env step (100 ms of sim time). The setting just multiplies the step counter shown in TensorBoard by 2,
so "step 210k" in logs means 105k real policy decisions.

Tuning knobs (batch size, train ratio, buffer size, eval cadence, etc.) live in the `PROFILES` dict in
`src/dreamliner/training/train.py`. Edit there if you need something off-menu.

Outputs land under `runs/dreamer/<run-name>/`. `latest.pt` is saved every 10 k env steps, on clean exit, **and on Ctrl+C
** (interrupts don't lose the model). `best.pt` is saved whenever eval score improves, with a `best.json` sidecar (step,
score, timestamp). `play` and `evaluate` load `best.pt` by default and fall back to `latest.pt`.

`--resume-from` loads weights only; optimizer state and replay buffer are rebuilt fresh. This is simpler than
snapshotting the ~12 GB GPU replay buffer and works fine for the "kill a run, extend it later" iteration loop.

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

Learning curves + per-scenario recovery stats over many rollouts. By default runs 200 greedy episodes against **both**
`best.pt` and `latest.pt`, re-seeding the env per checkpoint so they face the same scenario draws.

```bash
uv run evaluate                                        # latest run: plots + 200 rollouts per checkpoint
uv run evaluate runs/dreamer/baseline                  # specific run
uv run evaluate --episodes 500                         # bigger eval sample
uv run evaluate --checkpoint best                      # only best.pt
uv run evaluate --checkpoint latest                    # only latest.pt
uv run evaluate --no-eval                              # plots only, skip rollouts
```

Outputs land under `<logdir>/analysis/`:

- `learning_curves.png`: eval/train scalars from TensorBoard (checkpoint-agnostic)
- `recovery_metrics_<ckpt>.png`: altitude-loss / length / success-rate histograms per scenario
- `eval_episodes_<ckpt>.json`: raw per-episode trajectory log

### FlightGear

Standalone FG launcher. Use when you want the cockpit up before running `play --flightgear --no-fg-launch`, or just to
verify the visual stack.

```bash
uv run flightgear                                      # launch FG with 737-300
uv run flightgear --aircraft c172p                     # Cessna 172P visual instead
```

JSBSim drives the actual flight dynamics; the FG aircraft is only what you see on screen. Chase-cam (`v` in FG) is the
clearest "watch the recovery maneuver" view; cockpit (`ctrl+v` to cycle) gives you the attitude indicator + airspeed
tape.

## FlightGear setup (one-time, ~5 min)

Install [FlightGear](https://www.flightgear.org/download/) separately (not a pip dependency). We auto-discover `fgfs`
across the standard install locations (`C:\Program Files\FlightGear*\bin\fgfs.exe`, `/usr/games/fgfs`,
`/Applications/FlightGear.app/...`). No need to add it to PATH.

The default FG install only ships with the Cessna 172P; the 737-300 visual has to be downloaded once:

1. Open the FlightGear launcher (run `fgfs.exe` with no arguments, or use the Start menu shortcut).
2. Click the **Aircraft** tab on the left sidebar.
3. Click **Add-on Hangars** at the bottom of the aircraft list.
4. Make sure the **FlightGear 2024.1 Aircraft** hangar is checked (it should already be, since FG 2024 ships with it
   pre-configured pointing at `https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/catalog.xml`). Click **Update
   All** to refresh the catalog.
5. Back in the Aircraft list, type `737-300` in the search box at the top.
6. Click the 737-300 entry, then **Install** in the right pane. Wait for the download (~50 MB).
7. Once installed, close the launcher.

If FG 2024.1's catalog isn't pre-configured for some reason: in step 4, click **Add hangar** and paste
`https://mirrors.ibiblio.org/flightgear/ftp/Aircraft-2024/catalog.xml`, then continue with step 5.

## Environment

`StallRecoveryEnv` initialises the 737 in one of eight scenarios per episode and asks the policy to keep it flying (or
get it back to flying). Four "safe" scenarios teach the component skills that stall recovery needs; the other four (plus
the original `incipient_spin`) are the actual upset / recovery tasks.

**Scenarios** (defined in `configs/default.yaml`):

Safe scenarios, used early in training to teach recovery skills in a regime where the plane can't actually stall:

| name             | description                                      |
|------------------|--------------------------------------------------|
| `cruise`         | wings-level at 220-260 KCAS, just hold it there  |
| `gentle_turn`    | ±30° bank, agent rolls out to wings-level        |
| `pitch_recovery` | nose-up 10-25° with elevated but sub-stall alpha |
| `slow_flight`    | 140-180 KCAS in level flight, low-speed handling |

Upset scenarios (the actual recovery tasks):

| name                | description                                                |
|---------------------|------------------------------------------------------------|
| `high_alpha_entry`  | nose-up attitude, decaying airspeed, approaching stall AoA |
| `wings_level_stall` | classic low-speed wings-level stall                        |
| `turning_stall`     | banked-turn stall (20-60° bank)                            |
| `nose_high_upset`   | extreme pitch (30-60°) with low airspeed                   |
| `incipient_spin`    | high AoA with significant yaw rate                         |

Initial conditions (altitude, airspeed, alpha, pitch, roll, beta, yaw rate, throttle) are sampled per-episode from
per-scenario ranges in the YAML. See the [Curriculum](#curriculum) section below for how sampling weights shift across
training.

**Observation** (Dict, with a single `state` key for `mlp_keys=state` on Dreamer):

```text
state       : np.ndarray, shape (15,), dtype float32
is_first    : bool
is_last     : bool
is_terminal : bool
```

The 15-dim normalized state vector: alpha, beta, pitch, roll, calibrated airspeed, altitude, body angular rates (p, q,
r), normal load factor, throttle, elevator/aileron/rudder positions, vertical speed.

**Action** (`Box(-1, 1, shape=(4,))`): elevator, aileron, rudder, throttle. Throttle is internally remapped from [-1, 1]
to [0, 1] before being sent to JSBSim.

**Reward** (per agent step unless marked terminal):

- penalty for alpha above the safe threshold
- small per-step penalty for altitude lost from episode start (shaping signal)
- penalty for roll away from wings-level
- bonus for airspeed above stall speed
- penalty for jerky control inputs (smoothness regularizer)
- **terminal** one-shot penalty at episode end proportional to final ft lost from start (dominant altitude-loss cost)
- **terminal** one-shot **-100** on crash
- **terminal** one-shot **+500** when stable flight is held for 5 s

All weights live in `configs/default.yaml` under `reward:` and `targets:`.

**Crash** is triggered when the aircraft drops below the 1000 ft ground floor while descending, or when JSBSim itself
halts (numerical failure, ground impact).

**Success** requires holding a stable state for 5 s. The stability test checks alpha < safe threshold, wings roughly
level, airspeed above stall, descent rate bounded, **and** cumulative altitude loss below a scenario-dependent budget:
`max(2000 ft, 0.25 × start_altitude)`. The budget term keeps high-altitude starts from being trivially easy while the
absolute floor prevents low-altitude starts from being impossible.

**Termination**: crash, sustained recovery (5 s), or 60 s timeout.

**Timing**: JSBSim integrates at 120 Hz; the agent acts at 10 Hz, so each `env.step()` runs 12 JSBSim substeps.

## Curriculum

Before we added curriculum learning, every episode was a stall or an upset. The agent had to learn "how to fly" and "how
to recover from bad states" at the same time. A uniform-sampling 250 k run peaked at +306 eval with spotty per-scenario
performance (`turning_stall` was 0/1 in a spot-check; `nose_high_upset` was 1/2).

The curriculum splits training into four phases so the agent learns in a sensible order: hold stable flight, practice
the maneuvers that recovery actually needs, then do recovery.

| phase | env steps     | what the agent sees              | what it learns                                   |
|-------|---------------|----------------------------------|--------------------------------------------------|
| 0     | 0 - 20 k      | `cruise` only                    | basic stability and positive reward signal       |
| 1     | 20 k - 60 k   | all 4 safe scenarios             | roll-out, forward-stick, low-speed control       |
| 2     | 60 k - 170 k  | full mix, upsets ~83% of samples | stall recovery; safe skills stay in replay       |
| 3     | 170 k - 250 k | weighted toward hardest upsets   | specialize on `turning_stall`, `nose_high_upset` |

Each safe scenario targets one component of stall recovery:

- `cruise` is the equilibrium baseline. The main thing it teaches is that doing nothing aggressive is fine when nothing
  is wrong.
- `gentle_turn` starts the plane banked up to 30° in either direction. Agent has to roll out to wings-level. Precursor
  to `turning_stall` minus the stall.
- `pitch_recovery` starts the plane nose-up at 10 to 25° with alpha elevated but below stall AoA. The forward-stick
  reflex is the single most important stall-recovery input; this practices it in conditions where the plane won't
  actually stall.
- `slow_flight` runs at 140 to 180 KCAS in otherwise calm conditions. There's a big airspeed gap between cruise
  scenarios (220+ KCAS) and stall scenarios (80-180 KCAS). Without `slow_flight` the agent would hit stall-regime
  airspeeds for the first time during an actual upset, which is a bad time to be figuring out low-speed control.

### How it works

The trainer writes the current global step to `runs/<run>/_curriculum_step.txt` every 5 k steps (atomic replace, so env
subprocesses never read a half-written file). Every env reads that file on episode reset and picks weights from
whichever phase is active. One shared file, no IPC, no subprocess synchronization. Works on Windows spawn without any
special handling.

### Knobs

Set `curriculum.enabled: false` in `configs/default.yaml` for uniform sampling (the pre-curriculum baseline). The
`curriculum.phases` block takes a `start_step` and a `weights` dict per phase. Step boundaries are tuned for the `quick`
profile (250 k); for `good` (1 M), scale the boundaries by about 4x. Weights are renormalized per phase, so only
relative values matter; a weight of 0 excludes that scenario from the phase entirely.

## Configuration

`src/dreamliner/configs/default.yaml` controls scenarios, reward weights, recovery targets, and observation
normalization scales. The env loads this file by default; pass a different file or a Python dict to
`StallRecoveryEnv(config=...)` to override.

DreamerV3 model + trainer hyperparameters are pulled from `vendor/r2dreamer/configs/model/_base_.yaml` and the
per-profile `size{12M,25M,...,400M}.yaml` file, then overridden in `train.py:build_config` with our env-side knobs.

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

- **JSBSim XML input sockets**: the bundled `737.xml` declares two listening input sockets (telnet 5137, QTJSBSIM 5139).
  With multiple parallel envs (or any other process where one already binds them), subsequent FDMs spam
  `Could not bind to TCP/UDP input socket`. We copy the aircraft dir to a per-process temp location and strip those
  declarations once. See `dreamliner.utils.jsbsim_utils:_patched_aircraft_path`.
- **737 model approximation**: JSBSim's bundled 737 is built from public data, not Boeing proprietary. Stall behaviour
  is approximate but adequate for research. Swap to `c172p` in the YAML if you want better-validated stall
  characteristics.
- **Aileron position**: the 737 model exposes `fcs/left-aileron-pos-norm` and `fcs/right-aileron-pos-norm` but not a
  combined `fcs/aileron-pos-norm`. The env averages them.
- **Engine startup**: every reset calls `fdm.get_propulsion().init_running(-1)` so thrust is available immediately
  rather than spooling up over several seconds.
- **JSBSim reset**: in-place reset across very different ICs is unreliable, so the env rebuilds `FGFDMExec` on every
  `reset()`. Construction is cheap (sub-second).
- **JAX on Windows**: not an option for GPU. JAX has no Windows GPU wheels (cu12 or cu13). The official
  `danijar/dreamerv3` would require WSL2; we use NM512's PyTorch reproduction instead.
- **torch.compile on Windows**: enabled by default (`model.compile = True` in `build_config`). r2dreamer wraps
  `_cal_grad` with `torch.compile(mode="reduce-overhead")`; expected wall-clock improvement is ~30-50% once the graph is
  warm. First compile is slow (1-5 min cold, no tqdm progress during that window) and caches under
  `%USERPROFILE%\.triton\cache` for subsequent runs. Prereqs are `triton-windows` (pulled in by `uv sync`) and VS 2022
  Build Tools; `train.py` auto-sources `vcvars64.bat` so you don't need to launch from a Developer Command Prompt. If
  the cache path is a problem on constrained home dirs, set `TRITON_CACHE_DIR=C:\triton-cache` before running. Set
  `model.compile = False` to disable for debugging.
- **`uv run` rebuild lock on Windows**: `uv run train` and `uv add` rebuild the project wheel, which tries to replace
  `train.exe` in the venv. If another `train` is still running, that file is locked and the invocation errors with
  `The process cannot access the file because it is being used by another process`. Fix: Ctrl+C the running training (
  the next `latest.pt` has already been saved), then re-run.

## What is coming next

1. Latent-space visualization (t-SNE / UMAP of RSSM stoch+deter colored by stall phase). The "what did the world model
   learn" research figure.
2. Per-scenario eval logging during training (not just aggregate eval score) so the curriculum can be tuned on real data
   instead of a 5-episode play rollout.
3. Reward shaping experiments once we have per-scenario curves to compare.
