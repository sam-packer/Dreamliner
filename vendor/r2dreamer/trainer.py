import json
import logging
from collections import deque
from pathlib import Path
from statistics import median

import torch

import tools

log = logging.getLogger("dreamer.trainer")

_DIAGNOSTIC_WINDOW = 256
_EPISODE_DIAGNOSTIC_KEYS = (
    "success",
    "crash",
    "timeout",
    "altitude_loss_ft",
    "max_altitude_loss_ft",
    "max_abs_alpha_deg",
    "max_abs_roll_deg",
    "min_vc_kts",
    "max_descent_fps",
    "stall_fraction",
    "start_altitude_ft",
    "altitude_loss_budget_ft",
    "initial_alpha_deg",
    "initial_beta_deg",
    "initial_pitch_deg",
    "initial_roll_deg",
    "initial_vc_kts",
    "initial_yaw_rate_dps",
    "initial_throttle",
    "final_alpha_deg",
    "final_roll_deg",
    "final_vc_kts",
    "final_vspeed_fps",
    "stable_streak_steps",
    "episode_steps",
    "curriculum_step",
)
_WINDOW_MEAN_KEYS = (
    "score",
    "length",
    *_EPISODE_DIAGNOSTIC_KEYS,
    "reward_alpha",
    "reward_altitude_shape",
    "reward_roll",
    "reward_speed",
    "reward_smooth",
    "reward_alive",
    "reward_crash",
    "reward_success",
    "reward_altitude_terminal",
)
_WINDOW_MEDIAN_KEYS = (
    "score",
    "length",
    "altitude_loss_ft",
    "max_altitude_loss_ft",
    "max_abs_alpha_deg",
    "max_abs_roll_deg",
    "min_vc_kts",
    "max_descent_fps",
    "start_altitude_ft",
    "altitude_loss_budget_ft",
    "initial_beta_deg",
    "initial_pitch_deg",
    "initial_yaw_rate_dps",
    "initial_throttle",
    "final_alpha_deg",
    "final_roll_deg",
    "final_vc_kts",
    "final_vspeed_fps",
)
_SCENARIO_WINDOW_KEYS = (
    "success",
    "crash",
    "timeout",
    "score",
    "length",
    "altitude_loss_ft",
)


def _to_float(value):
    if isinstance(value, torch.Tensor):
        value = value.item()
    return float(value)


def _mean(rows, key):
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    return sum(values) / len(values)


def _median(rows, key):
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    return float(median(values))


def _log_scalar(logger, name, value):
    if value is not None:
        logger.scalar(name, value)


def _append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def _log_window_diagnostics(logger, episodes):
    rows = list(episodes)
    if not rows:
        return

    logger.scalar("train_window/episodes", len(rows))
    for key in _WINDOW_MEAN_KEYS:
        _log_scalar(logger, f"train_window/{key}", _mean(rows, key))
    for key in _WINDOW_MEDIAN_KEYS:
        _log_scalar(logger, f"train_window/{key}_median", _median(rows, key))

    curriculum_steps = [float(row["curriculum_step"]) for row in rows if "curriculum_step" in row]
    if curriculum_steps:
        logger.scalar("train_window/curriculum_step_min", min(curriculum_steps))
        logger.scalar("train_window/curriculum_step_max", max(curriculum_steps))

    # `scenario_name` is a human-readable label added alongside the numeric
    # one-hot `scenario_<name>` flags; exclude it from numeric aggregation.
    scenario_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("scenario_") and key != "scenario_name"
        }
    )
    total = float(len(rows))
    for scenario_key in scenario_keys:
        scenario_rows = [row for row in rows if row.get(scenario_key, 0.0) > 0.5]
        if not scenario_rows:
            continue
        scenario_name = scenario_key[len("scenario_"):]
        prefix = f"train_window/by_scenario/{scenario_name}"
        logger.scalar(f"{prefix}/episodes", len(scenario_rows))
        logger.scalar(f"{prefix}/share", len(scenario_rows) / total)
        for key in _SCENARIO_WINDOW_KEYS:
            _log_scalar(logger, f"{prefix}/{key}", _mean(scenario_rows, key))
        _log_scalar(logger, f"{prefix}/score_median", _median(scenario_rows, "score"))
        _log_scalar(
            logger,
            f"{prefix}/altitude_loss_ft_median",
            _median(scenario_rows, "altitude_loss_ft"),
        )


class OnlineTrainer:
    def __init__(self, config, replay_buffer, logger, logdir, train_envs, eval_envs,
                 save_fn=None, save_every=0, progress_fn=None, on_eval=None):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.pretrain = int(config.pretrain)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.video_pred_log = bool(config.video_pred_log)
        self.params_hist_log = bool(config.params_hist_log)
        self.batch_length = int(config.batch_length)
        batch_steps = int(config.batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio * config.action_repeat)
        self._should_pretrain = tools.Once()
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._action_repeat = config.action_repeat
        # Optional periodic checkpoint: save_fn(step, agent) is called every
        # `save_every` env steps during training.
        self._save_fn = save_fn
        self._should_save = tools.Every(int(save_every)) if save_every and save_every > 0 else None
        # Optional progress callback: progress_fn(step, last_metrics_dict) is
        # called once per main-loop iteration; train_dreamer uses it to drive a
        # tqdm bar without coupling trainer.py to tqdm.
        self._progress_fn = progress_fn
        # Optional post-evaluation callback: on_eval(step, mean_eval_score, agent)
        # runs after each evaluation phase so the training driver can track
        # the best-so-far policy and save best.pt aside from latest.pt.
        self._on_eval = on_eval
        self._episode_diag_path = Path(logdir) / "episode_diagnostics.jsonl"
        self._recent_episodes = deque(maxlen=_DIAGNOSTIC_WINDOW)

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Environment stepping is executed on CPU to avoid GPU<->CPU synchronizations
        in the worker processes. Observations are moved back to GPU asynchronously
        (H2D with non_blocking=True) right before policy inference.
        """
        log.info("Evaluating policy...")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num, dtype=torch.bool, device=agent.device)
        steps = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        log_metrics = {}
        # cache is only used for video logging / open-loop prediction.
        cache = []
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments on CPU.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Store transition.
            # We keep the observation and the action that produced it together.
            trans["action"] = act
            if len(cache) < self.batch_length:
                # Terminal diagnostics are sparse, so keep them out of the
                # video/open-loop cache to preserve a stackable key set.
                log_keys = [key for key in trans.keys() if key.startswith("log_")]
                cache.append((trans.exclude(*log_keys) if log_keys else trans).clone())
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        # dict of (B, T, *)
        cache = torch.stack(cache, dim=1) if len(cache) else None
        mean_eval_score = float(returns.mean().item())
        self.logger.scalar("episode/eval_score", mean_eval_score)
        self.logger.scalar("episode/eval_length", steps.to(torch.float32).mean())
        if self._on_eval is not None:
            self._on_eval(train_step, mean_eval_score, agent)
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value, max=1.0)  # make sure 1.0 for success episode
            self.logger.scalar(f"episode/eval_{key[4:]}", value.mean())
        if cache is not None and "image" in cache:
            self.logger.video("eval_video", tools.to_np(cache["image"][:1]))
        if self.video_pred_log and cache is not None:
            initial = agent.get_initial_state(1)
            self.logger.video(
                "eval_open_loop",
                tools.to_np(
                    agent.video_pred(
                        cache[:1],  # give only first batch
                        (initial["stoch"], initial["deter"]),
                    )
                ),
            )
        self.logger.write(train_step)
        agent.train()

    def begin(self, agent, start_step=0, update_count=0):
        """Main online training loop.

        The loop is designed to overlap CPU environment stepping and GPU model
        execution. Environments are stepped on CPU, observations are pinned,
        then transferred to GPU with non_blocking=True.
        """
        envs = self.train_envs
        video_cache = []
        step = int(start_step)
        update_count = int(update_count)
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num, dtype=torch.float32, device=agent.device)
        lengths = torch.zeros(envs.env_num, dtype=torch.int32, device=agent.device)
        episode_ids = torch.arange(
            envs.env_num, dtype=torch.int32, device=agent.device
        )  # Increment this to prevent sampling across episode boundaries
        next_episode_id = int(envs.env_num)
        episode_metrics = {}
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        if i == 0 and len(video_cache) > 0:
                            video = torch.stack(video_cache, axis=0)
                            self.logger.video("train_video", tools.to_np(video[None]))
                            video_cache = []
                        episode = {
                            "score": _to_float(returns[i]),
                            "length": _to_float(lengths[i]),
                        }
                        for name, values in episode_metrics.items():
                            episode[name] = _to_float(values[i])
                        scenario_name = None
                        for name, value in episode.items():
                            if name.startswith("scenario_") and value > 0.5:
                                scenario_name = name[len("scenario_"):]
                                break
                        if scenario_name is not None:
                            episode["scenario_name"] = scenario_name
                        _append_jsonl(self._episode_diag_path, {"step": int(step + i), **episode})
                        self._recent_episodes.append(dict(episode))
                        self.logger.scalar("episode/score", episode["score"])
                        self.logger.scalar("episode/length", episode["length"])
                        for name in _EPISODE_DIAGNOSTIC_KEYS:
                            if name in episode:
                                self.logger.scalar(f"episode/{name}", episode[name])
                        self.logger.write(step + i)  # to show all values on tensorboard
                        returns[i] = lengths[i] = 0
                        for values in episode_metrics.values():
                            values[i] = 0.0
                        episode_ids[i] = next_episode_id
                        next_episode_id += 1
            step += int((~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done
            if self._progress_fn is not None:
                self._progress_fn(step)

            # Step environments on CPU to avoid GPU<->CPU sync in the worker processes.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            act, agent_state = agent.act(trans.clone(), agent_state, eval=False)

            # Store transition.
            # We keep the observation and the action that produced it together.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            trans["stoch"] = agent_state["stoch"]
            trans["deter"] = agent_state["deter"]
            trans["episode"] = episode_ids  # Don't lift dim
            if "image" in trans:
                video_cache.append(trans["image"][0])
            self.replay_buffer.add_transition(trans.detach())
            returns += trans["reward"][:, 0]
            for key, value in trans.items():
                if key.startswith("log_"):
                    metric_name = key[4:]
                    if metric_name not in episode_metrics:
                        episode_metrics[metric_name] = torch.zeros(
                            envs.env_num,
                            dtype=torch.float32,
                            device=agent.device,
                        )
                    episode_metrics[metric_name] += value[:, 0]
            # Update models after enough data has accumulated
            if self.replay_buffer.count() > envs.env_num * (self.batch_length + 1):
                if self._should_pretrain():
                    update_num = self.pretrain
                else:
                    update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    for name, value in train_metrics.items():
                        value = tools.to_np(value) if isinstance(value, torch.Tensor) else value
                        self.logger.scalar(f"train/{name}", value)
                    self.logger.scalar("train/opt/updates", update_count)
                    _log_window_diagnostics(self.logger, self._recent_episodes)
                    if self.video_pred_log:
                        data, _, initial = self.replay_buffer.sample()
                        self.logger.video("open_loop", tools.to_np(agent.video_pred(data, initial)))
                    if self.params_hist_log:
                        for name, param in agent._named_params.items():
                            self.logger.histogram(name, tools.to_np(param))
                    self.logger.write(step, fps=True)
            # Optional periodic checkpoint save.
            if self._save_fn is not None and self._should_save is not None and self._should_save(step):
                self._save_fn(step, agent)
