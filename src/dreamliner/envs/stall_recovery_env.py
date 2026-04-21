"""Gymnasium environment: random commercial-aircraft stall, agent must recover.

Observations are returned as a Dict with a single ``state`` key so the same env
can drive DreamerV3 (`encoder.mlp_keys='state'`) and SB3 ``MultiInputPolicy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from dreamliner.utils import jsbsim_utils as J
from dreamliner.utils.jsbsim_utils import P, ScenarioSpec


# Order of the flat state vector. Must match _read_state() below.
STATE_KEYS: tuple[str, ...] = (
    "alpha_norm",
    "beta_norm",
    "pitch_norm",
    "roll_norm",
    "vc_norm",
    "altitude_norm",
    "p_norm",
    "q_norm",
    "r_norm",
    "load_factor",
    "throttle",
    "elevator_pos",
    "aileron_pos",
    "rudder_pos",
    "vspeed_norm",
)
STATE_DIM = len(STATE_KEYS)


class StallRecoveryEnv(gym.Env):
    """Gymnasium env wrapping JSBSim for commercial-aircraft stall recovery."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict | str | Path | None = None,
        *,
        flightgear: bool = False,
        curriculum_step_file: str | Path | None = None,
        disable_curriculum: bool = False,
    ):
        super().__init__()

        cfg = _load_config(config)
        env_cfg = cfg["env"]
        self._output_directive = J.flightgear_directive_path() if flightgear else None

        self._aircraft: str = env_cfg["aircraft"]
        self._sim_dt_hz: int = int(env_cfg["sim_dt_hz"])
        self._agent_dt_hz: int = int(env_cfg["agent_dt_hz"])
        if self._sim_dt_hz % self._agent_dt_hz != 0:
            raise ValueError("sim_dt_hz must be a multiple of agent_dt_hz")
        self._substeps: int = self._sim_dt_hz // self._agent_dt_hz
        self._max_episode_steps: int = int(env_cfg["max_episode_seconds"] * self._agent_dt_hz)
        self._ground_floor_ft: float = float(env_cfg["ground_floor_ft"])
        self._success_hold_steps: int = int(env_cfg["success_hold_seconds"] * self._agent_dt_hz)

        self._scenarios: list[ScenarioSpec] = J.parse_scenarios(cfg["scenarios"])
        self._targets: dict = cfg["targets"]
        self._reward: dict = cfg["reward"]
        self._scales: dict = cfg["obs_scales"]

        curriculum_enabled, curriculum_phases = J.parse_curriculum(cfg.get("curriculum"))
        self._curriculum: J.CurriculumSchedule | None = None
        if curriculum_enabled and not disable_curriculum:
            step_file = Path(curriculum_step_file) if curriculum_step_file else None
            self._curriculum = J.CurriculumSchedule(curriculum_phases, step_file)

        # Action: elevator, aileron, rudder, throttle, all in [-1, 1].
        # Throttle is remapped to [0, 1] internally.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )
        self.observation_space = spaces.Dict({
            "state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(STATE_DIM,), dtype=np.float32,
            ),
        })

        self._fdm = None
        self._np_random: np.random.Generator | None = None
        self._step_idx: int = 0
        self._start_altitude_ft: float = 0.0
        self._episode_curriculum_step: int = 0
        self._stable_streak: int = 0
        self._prev_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self._current_scenario_name: str = ""
        self._scenario_names: tuple[str, ...] = tuple(s.name for s in self._scenarios)
        self._episode_initial_alpha_deg: float = 0.0
        self._episode_initial_beta_deg: float = 0.0
        self._episode_initial_pitch_deg: float = 0.0
        self._episode_initial_roll_deg: float = 0.0
        self._episode_initial_vc_kts: float = 0.0
        self._episode_initial_yaw_rate_dps: float = 0.0
        self._episode_initial_throttle: float = 0.0
        self._episode_altitude_loss_budget_ft: float = 0.0
        self._episode_max_abs_alpha_deg: float = 0.0
        self._episode_max_abs_roll_deg: float = 0.0
        self._episode_min_vc_kts: float = 0.0
        self._episode_max_descent_fps: float = 0.0
        self._episode_max_altitude_loss_ft: float = 0.0
        self._episode_stall_steps: int = 0
        self._episode_reward_terms: dict[str, float] = {}

    # --- Gym API ------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None or self._np_random is None:
            self._np_random = np.random.default_rng(seed)

        # JSBSim has known issues being reset in-place across very different ICs;
        # the safest reliable pattern is to rebuild the FDM each episode.
        self._fdm = J.make_fdm(self._aircraft, self._sim_dt_hz, self._output_directive)

        forced_scenario = None
        if options is not None:
            scenario_name = options.get("scenario")
            if scenario_name is not None:
                forced_scenario = next(
                    (scenario for scenario in self._scenarios if scenario.name == scenario_name),
                    None,
                )
                if forced_scenario is None:
                    raise ValueError(f"unknown scenario override: {scenario_name}")

        if forced_scenario is not None:
            scenario = forced_scenario
            curriculum_step = self._curriculum.current_step() if self._curriculum is not None else 0
        elif self._curriculum is not None:
            scenario, curriculum_step = self._curriculum.sample(self._np_random, self._scenarios)
        else:
            scenario = J.sample_scenario(self._np_random, self._scenarios)
            curriculum_step = 0
        self._current_scenario_name = scenario.name
        ics = J.apply_scenario(self._fdm, scenario, self._np_random)

        self._step_idx = 0
        self._episode_curriculum_step = int(curriculum_step)
        self._stable_streak = 0
        self._start_altitude_ft = float(self._fdm[P.altitude_ft])
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._reset_episode_diagnostics()
        initial_raw = self._read_state_raw()
        self._episode_initial_beta_deg = initial_raw["beta_deg"]
        self._episode_initial_pitch_deg = float(ics["pitch_deg"])
        self._episode_initial_alpha_deg = initial_raw["alpha_deg"]
        self._episode_initial_roll_deg = float(np.degrees(initial_raw["roll_rad"]))
        self._episode_initial_vc_kts = initial_raw["vc_kts"]
        self._episode_initial_yaw_rate_dps = float(ics["yaw_rate_dps"])
        self._episode_initial_throttle = float(ics["throttle"])
        self._episode_altitude_loss_budget_ft = self._max_altitude_loss_budget_ft()
        self._update_episode_diagnostics(initial_raw)

        obs = self._read_obs()
        info = {"scenario": scenario.name, "ics": ics, "curriculum_step": curriculum_step}
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(4)
        action = np.clip(action, -1.0, 1.0)

        self._fdm[P.elevator_cmd] = float(action[0])
        self._fdm[P.aileron_cmd]  = float(action[1])
        self._fdm[P.rudder_cmd]   = float(action[2])
        self._fdm[P.throttle_cmd] = float((action[3] + 1.0) * 0.5)  # [-1,1] -> [0,1]

        # Run sim_dt_hz / agent_dt_hz integration steps per agent step. If
        # fdm.run() ever returns False (ground impact or numerical failure),
        # treat the episode as crashed: otherwise the loop below would just
        # keep polling a dead sim until the time-limit truncation fires.
        sim_failed = False
        for _ in range(self._substeps):
            if not self._fdm.run():
                sim_failed = True
                break

        obs = self._read_obs()
        raw = self._read_state_raw()

        below_floor = raw["altitude_ft"] < self._ground_floor_ft and raw["vspeed_fps"] > 0.0
        crashed = sim_failed or below_floor
        recovered = self._is_recovered(raw)
        if recovered:
            self._stable_streak += 1
        else:
            self._stable_streak = 0
        success = self._stable_streak >= self._success_hold_steps
        self._update_episode_diagnostics(raw)

        reward, reward_terms = self._compute_reward(raw, action, crashed=crashed, success=success)

        self._step_idx += 1
        terminated = bool(crashed or success)
        truncated = bool(self._step_idx >= self._max_episode_steps)

        # Terminal altitude-loss penalty: one lump at episode end proportional
        # to total ft lost from start. Keeps the per-step alt term small (pure
        # gradient signal) while the terminal term carries the real cost of a
        # deep recovery. Applied on success, crash, and timeout.
        if terminated or truncated:
            final_alt_loss = max(0.0, self._start_altitude_ft - raw["altitude_ft"])
            terminal_altitude_term = -self._reward["altitude_loss_terminal_penalty_per_ft"] * final_alt_loss
            reward += terminal_altitude_term
            reward_terms["altitude_terminal"] = terminal_altitude_term
        else:
            reward_terms["altitude_terminal"] = 0.0
        self._accumulate_reward_terms(reward_terms)

        self._prev_action = action
        timeout = bool(truncated and not terminated)
        outcome = "success" if success else "crash" if crashed else "timeout" if timeout else "running"

        info = {
            "scenario": self._current_scenario_name,
            "outcome": outcome,
            "altitude_ft": raw["altitude_ft"],
            "altitude_loss_ft": self._start_altitude_ft - raw["altitude_ft"],
            "alpha_deg": raw["alpha_deg"],
            "vc_kts": raw["vc_kts"],
            "roll_deg": np.degrees(raw["roll_rad"]),
            "stalled": raw["stall_hyst_norm"] > 0.5,
            "crashed": crashed,
            "success": success,
            "stable_streak_steps": self._stable_streak,
        }
        if terminated or truncated:
            info.update({
                "log_success": float(success),
                "log_crash": float(crashed),
                "log_timeout": float(timeout),
                "log_altitude_loss_ft": float(info["altitude_loss_ft"]),
                "log_max_altitude_loss_ft": float(self._episode_max_altitude_loss_ft),
                "log_max_abs_alpha_deg": float(self._episode_max_abs_alpha_deg),
                "log_max_abs_roll_deg": float(self._episode_max_abs_roll_deg),
                "log_min_vc_kts": float(self._episode_min_vc_kts),
                "log_max_descent_fps": float(self._episode_max_descent_fps),
                "log_stall_fraction": float(self._episode_stall_steps / max(1, self._step_idx)),
                "log_curriculum_step": float(self._episode_curriculum_step),
                "log_start_altitude_ft": float(self._start_altitude_ft),
                "log_altitude_loss_budget_ft": float(self._episode_altitude_loss_budget_ft),
                "log_initial_alpha_deg": float(self._episode_initial_alpha_deg),
                "log_initial_beta_deg": float(self._episode_initial_beta_deg),
                "log_initial_pitch_deg": float(self._episode_initial_pitch_deg),
                "log_initial_roll_deg": float(self._episode_initial_roll_deg),
                "log_initial_vc_kts": float(self._episode_initial_vc_kts),
                "log_initial_yaw_rate_dps": float(self._episode_initial_yaw_rate_dps),
                "log_initial_throttle": float(self._episode_initial_throttle),
                "log_final_alpha_deg": float(raw["alpha_deg"]),
                "log_final_roll_deg": float(np.degrees(raw["roll_rad"])),
                "log_final_vc_kts": float(raw["vc_kts"]),
                "log_final_vspeed_fps": float(raw["vspeed_fps"]),
                "log_stable_streak_steps": float(self._stable_streak),
                "log_episode_steps": float(self._step_idx),
            })
            for name in self._scenario_names:
                info[f"log_scenario_{name}"] = float(name == self._current_scenario_name)
            for key, value in self._episode_reward_terms.items():
                info[f"log_reward_{key}"] = float(value)
        return obs, float(reward), terminated, truncated, info

    def close(self) -> None:
        # FGFDMExec doesn't need explicit teardown; drop the reference so the
        # underlying object is destroyed before any next reset rebuilds it.
        self._fdm = None

    # --- Observation / state ------------------------------------------------

    def _read_state_raw(self) -> dict[str, float]:
        f = self._fdm
        return {
            "alpha_deg":       float(f[P.alpha_deg]),
            "beta_deg":        float(f[P.beta_deg]),
            "stall_hyst_norm": float(f[P.stall_hyst_norm]),
            "pitch_rad":       float(f[P.pitch_rad]),
            "roll_rad":        float(f[P.roll_rad]),
            "vc_kts":          float(f[P.vc_kts]),
            "altitude_ft":     float(f[P.altitude_ft]),
            "p_rad_sec":       float(f[P.p_rad_sec]),
            "q_rad_sec":       float(f[P.q_rad_sec]),
            "r_rad_sec":       float(f[P.r_rad_sec]),
            "load_factor_g":   float(f[P.load_factor_g]),
            "throttle":        float(f[P.throttle_cmd]),
            "elevator_pos":    float(f[P.elevator_pos_norm]),
            "aileron_pos":     float(J.aileron_pos_norm(f)),
            "rudder_pos":      float(f[P.rudder_pos_norm]),
            "vspeed_fps":      float(f[P.v_down_fps]),  # +ve = descending
        }

    def _read_obs(self) -> dict[str, np.ndarray]:
        s = self._read_state_raw()
        sc = self._scales
        vec = np.array([
            s["alpha_deg"]    / sc["alpha_deg"],
            s["beta_deg"]     / sc["beta_deg"],
            s["pitch_rad"]    / sc["pitch_rad"],
            s["roll_rad"]     / sc["roll_rad"],
            s["vc_kts"]       / sc["vc_kts"],
            s["altitude_ft"]  / sc["altitude_ft"],
            s["p_rad_sec"]    / sc["rate_rad_sec"],
            s["q_rad_sec"]    / sc["rate_rad_sec"],
            s["r_rad_sec"]    / sc["rate_rad_sec"],
            s["load_factor_g"]/ sc["load_factor_g"],
            s["throttle"]     / sc["throttle"],
            s["elevator_pos"] / sc["surface_pos"],
            s["aileron_pos"]  / sc["surface_pos"],
            s["rudder_pos"]   / sc["surface_pos"],
            s["vspeed_fps"]   / sc["vspeed_fps"],
        ], dtype=np.float32)
        return {"state": vec}

    # --- Reward / termination ------------------------------------------------

    def _max_altitude_loss_budget_ft(self) -> float:
        t = self._targets
        return max(
            t["max_altitude_loss_ft"],
            t["max_altitude_loss_fraction_of_start"] * self._start_altitude_ft,
        )

    def _is_recovered(self, raw: dict[str, float]) -> bool:
        t = self._targets
        altitude_loss = max(0.0, self._start_altitude_ft - raw["altitude_ft"])
        return (
            raw["alpha_deg"] < t["alpha_deg_safe"]
            and abs(np.degrees(raw["roll_rad"])) < t["roll_deg_tolerance"]
            and raw["vc_kts"] > t["airspeed_kcas_min"]
            and -raw["vspeed_fps"] > t["vertical_speed_fps_min"]
            and altitude_loss < self._max_altitude_loss_budget_ft()
        )

    def _compute_reward(
        self,
        raw: dict[str, float],
        action: np.ndarray,
        *,
        crashed: bool,
        success: bool,
    ) -> tuple[float, dict[str, float]]:
        r = self._reward
        t = self._targets

        alpha_target_deg = min(t["alpha_deg_threshold"], t["alpha_deg_safe"])
        alpha_excess = max(0.0, raw["alpha_deg"] - alpha_target_deg)
        alpha_term = -r["alpha_penalty_per_deg"] * alpha_excess

        altitude_loss = max(0.0, self._start_altitude_ft - raw["altitude_ft"])
        alt_term = -r["altitude_loss_penalty_per_ft"] * altitude_loss

        roll_term = -r["roll_penalty_per_deg"] * abs(np.degrees(raw["roll_rad"]))

        speed_deficit = max(0.0, t["airspeed_kcas_min"] - raw["vc_kts"])
        speed_term = -r["airspeed_bonus_per_kt"] * speed_deficit

        delta = np.abs(action - self._prev_action).mean()
        smooth_term = -r["control_smoothness_penalty"] * float(delta)
        alive_term = r.get("step_alive_bonus", 0.0)
        crash_term = r["crash_penalty"] if crashed else 0.0
        success_term = r["success_bonus"] if success else 0.0

        total = (
            alpha_term + alt_term + roll_term + speed_term + smooth_term + alive_term
            + crash_term + success_term
        )
        return total, {
            "alpha": float(alpha_term),
            "altitude_shape": float(alt_term),
            "roll": float(roll_term),
            "speed": float(speed_term),
            "smooth": float(smooth_term),
            "alive": float(alive_term),
            "crash": float(crash_term),
            "success": float(success_term),
        }

    def _reset_episode_diagnostics(self) -> None:
        self._episode_max_abs_alpha_deg = 0.0
        self._episode_max_abs_roll_deg = 0.0
        self._episode_min_vc_kts = float("inf")
        self._episode_max_descent_fps = 0.0
        self._episode_max_altitude_loss_ft = 0.0
        self._episode_stall_steps = 0
        self._episode_reward_terms = {
            "alpha": 0.0,
            "altitude_shape": 0.0,
            "roll": 0.0,
            "speed": 0.0,
            "smooth": 0.0,
            "alive": 0.0,
            "crash": 0.0,
            "success": 0.0,
            "altitude_terminal": 0.0,
        }

    def _update_episode_diagnostics(self, raw: dict[str, float]) -> None:
        altitude_loss = max(0.0, self._start_altitude_ft - raw["altitude_ft"])
        self._episode_max_abs_alpha_deg = max(self._episode_max_abs_alpha_deg, abs(raw["alpha_deg"]))
        self._episode_max_abs_roll_deg = max(self._episode_max_abs_roll_deg, abs(np.degrees(raw["roll_rad"])))
        self._episode_min_vc_kts = min(self._episode_min_vc_kts, raw["vc_kts"])
        self._episode_max_descent_fps = max(self._episode_max_descent_fps, raw["vspeed_fps"])
        self._episode_max_altitude_loss_ft = max(self._episode_max_altitude_loss_ft, altitude_loss)
        if raw["stall_hyst_norm"] > 0.5:
            self._episode_stall_steps += 1

    def _accumulate_reward_terms(self, reward_terms: dict[str, float]) -> None:
        for key, value in reward_terms.items():
            self._episode_reward_terms[key] = self._episode_reward_terms.get(key, 0.0) + float(value)


# --- Config loading ----------------------------------------------------------

def _load_config(config: dict | str | Path | None) -> dict:
    if isinstance(config, dict):
        return config
    path = Path(config) if config else J.default_config_path()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
