"""Bridge from StallRecoveryEnv (Gymnasium 5-tuple) to r2dreamer's API.

r2dreamer (vendored under ``vendor/r2dreamer``) expects the legacy Gym v0.21
contract: ``reset() -> obs`` (no info), ``step(action) -> (obs, reward, done, info)``,
and observations carry ``is_first`` / ``is_last`` / ``is_terminal`` flag entries
in the obs dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from dreamliner.envs.stall_recovery_env import StallRecoveryEnv


class DreamerStallEnv:
    """Gym-v0.21-style wrapper around StallRecoveryEnv for r2dreamer."""

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        seed: int = 0,
        *,
        config: dict | str | Path | None = None,
        flightgear: bool = False,
        curriculum_step_file: str | Path | None = None,
        disable_curriculum: bool = False,
    ):
        self._env = StallRecoveryEnv(
            config=config,
            flightgear=flightgear,
            curriculum_step_file=curriculum_step_file,
            disable_curriculum=disable_curriculum,
        )
        self._seed = int(seed)
        self._reset_seed_used = False
        self.last_reset_info: dict[str, Any] = {}

    @property
    def observation_space(self) -> gym.spaces.Dict:
        spaces = dict(self._env.observation_space.spaces)
        spaces["is_first"]    = gym.spaces.Box(0, 1, shape=(), dtype=bool)
        spaces["is_last"]     = gym.spaces.Box(0, 1, shape=(), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, shape=(), dtype=bool)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self._env.action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        # Apply the deterministic seed only the first time so each fresh episode
        # samples a different scenario via the env's internal RNG.
        if seed is None:
            seed = self._seed if not self._reset_seed_used else None
        self._reset_seed_used = True
        obs, info = self._env.reset(seed=seed, options=options)
        self.last_reset_info = dict(info)
        return self._add_flags(obs, is_first=True, is_last=False, is_terminal=False)

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = bool(terminated or truncated)
        obs = self._add_flags(obs, is_first=False, is_last=done, is_terminal=bool(terminated))
        return obs, float(reward), done, info

    def close(self) -> None:
        self._env.close()

    @staticmethod
    def _add_flags(
        obs: dict[str, np.ndarray],
        *,
        is_first: bool,
        is_last: bool,
        is_terminal: bool,
    ) -> dict[str, np.ndarray]:
        out = {k: np.asarray(v, dtype=np.float32) for k, v in obs.items()}
        out["is_first"]    = np.asarray(is_first,    dtype=bool)
        out["is_last"]     = np.asarray(is_last,     dtype=bool)
        out["is_terminal"] = np.asarray(is_terminal, dtype=bool)
        return out
