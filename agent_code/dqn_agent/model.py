from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from .replay_buffer import DictReplayBuffer
from .schedules import LinearSchedule
from .symmetry import TensorAugmenter

__all__ = ["BombermanFeatureExtractor", "QNetwork", "MaskableDQN"]


class StepProfiler:
    """Opt-in timing breakdown of train-step stages using a rolling window."""

    def __init__(self, window: int = 50):
        self._times: dict[str, deque] = {}
        self._t0: dict[str, float] = {}
        self.window = window

    def start(self, name: str) -> None:
        self._t0[name] = time.perf_counter()

    def stop(self, name: str, device: Optional[torch.device] = None) -> None:
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
        dt_ms = (time.perf_counter() - self._t0[name]) * 1000.0
        bucket = self._times.setdefault(name, deque(maxlen=self.window))
        bucket.append(dt_ms)

    def summary(self) -> dict[str, float]:
        """Mean milliseconds per stage over the rolling window."""
        return {name: (sum(vals) / len(vals)) for name, vals in self._times.items() if vals}


def _dropout_layer(dropout: float) -> list:
    return [nn.Dropout(dropout)] if dropout > 0 else []


class ResidualBlock(nn.Module):
    """A residual CNN block with group normalization."""

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.block(x))


class BombermanFeatureExtractor(nn.Module):
    """Encode board and feature vectors into a latent representation."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.features_dim = features_dim

        grid_space = observation_space["grid_tensor"]
        feature_space = observation_space["features"]

        n_input_channels = grid_space.shape[0]
        h, w = grid_space.shape[1], grid_space.shape[2]

        stem_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, stem_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(stem_channels),
            ResidualBlock(stem_channels),
        )

        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, h, w, dtype=torch.float32)
            n_cnn_flatten = self._conv_forward(sample).shape[1]

        self.grid_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten, 128),
            nn.ReLU(),
            *_dropout_layer(dropout),
        )

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU(),
            *_dropout_layer(dropout),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
            *_dropout_layer(dropout),
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the CNN feature stack over a tensor."""
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.flatten(x)

    def forward(self, observations: dict) -> torch.Tensor:
        """Convert a Bomberman observation dict into a feature vector."""
        grid_tensor = observations["grid_tensor"].float()
        features = observations["features"].float()

        cnn_out = self._conv_forward(grid_tensor)
        cnn_fc_out = self.grid_fc(cnn_out)

        features_fc_out = self.features_preprocess_fc(features)

        combined = torch.cat([cnn_fc_out, features_fc_out], dim=1)
        return self.combined_fc(combined)


class QNetwork(nn.Module):
    """Feature extractor plus Q-value head for discrete actions."""

    def __init__(self, observation_space: gym.spaces.Dict, n_actions: int, features_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.features_extractor = BombermanFeatureExtractor(observation_space, features_dim, dropout)
        self.q_head = nn.Linear(features_dim, n_actions)

    def forward(self, observations: dict) -> torch.Tensor:
        """Return Q-values for each legal action."""
        return self.q_head(self.features_extractor(observations))


def _obs_to_tensors(observation: dict, device: torch.device, pinned: Optional[dict] = None) -> dict:
    """Convert observation arrays to PyTorch tensors on a device."""
    grid_np = np.asarray(observation["grid_tensor"])
    feats_np = np.asarray(observation["features"])
    non_blocking = device.type == "cuda"

    if non_blocking and pinned is not None:
        pinned["grid"].copy_(torch.from_numpy(grid_np))
        pinned["features"].copy_(torch.from_numpy(feats_np))
        grid_t = pinned["grid"]
        feats_t = pinned["features"]
    else:
        grid_t = torch.as_tensor(grid_np, dtype=torch.float32)
        feats_t = torch.as_tensor(feats_np, dtype=torch.float32)

    return {
        "grid_tensor": grid_t.to(device, non_blocking=non_blocking),
        "features": feats_t.to(device, non_blocking=non_blocking),
    }


def _is_vectorized(observation: dict, observation_space: gym.spaces.Dict) -> bool:
    """Check whether an observation already contains a batch dimension."""
    feats = np.asarray(observation["features"])
    return feats.ndim > len(observation_space["features"].shape)


class MaskableDQN:
    """A compact DQN implementation with action masking."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space,
        *,
        learning_rate: float = 1e-4,
        buffer_size: int = 500_000,
        learning_starts: int = 50_000,
        batch_size: int = 256,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: int = 4,
        gradient_steps: int = 1,
        target_update_interval: int = 10_000,
        exploration_fraction: float = 0.3,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10.0,
        features_dim: int = 256,
        exploration_duration: int = 300_000,
        symmetry_augmentation: bool = False,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        n_envs: int = 1,
        device: str = "cpu",
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.n_actions = int(action_space.n)
        self.n_envs = max(1, int(n_envs))
        self.device = torch.device(device)

        self.learning_rate = learning_rate
        self.buffer_size = int(buffer_size)
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.tau = tau
        self.gamma = gamma
        self.train_freq = max(1, int(train_freq))
        self.gradient_steps = int(gradient_steps)
        self.target_update_interval = max(1, int(target_update_interval))
        self.max_grad_norm = max_grad_norm
        self.features_dim = features_dim
        self.symmetry_augmentation = bool(symmetry_augmentation)
        self._augmenter = TensorAugmenter(self.device)
        self.weight_decay = float(weight_decay)
        self.dropout = float(dropout)

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_duration = max(1, int(exploration_duration))
        self.exploration_schedule = LinearSchedule(
            exploration_initial_eps, exploration_final_eps, self.exploration_duration,
        )
        self.exploration_rate = exploration_initial_eps

        self.q_net = QNetwork(observation_space, self.n_actions, features_dim, self.dropout).to(self.device)
        self.q_net_target = QNetwork(observation_space, self.n_actions, features_dim, self.dropout).to(self.device)
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        self.q_net_target.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self._amp_enabled = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._amp_enabled)

        self.replay_buffer: Optional[DictReplayBuffer] = None

        self.num_timesteps = 0
        self._steps_since_train = 0
        self._last_obs = None
        self._last_action_masks = None
        self.last_loss_mean: Optional[float] = None
        self.n_updates = 0
        self.last_grad_norm: Optional[float] = None
        self.last_mean_q: Optional[float] = None
        self.last_td_error: Optional[float] = None
        self._loss_window: deque = deque(maxlen=100)
        self._grad_norm_window: deque = deque(maxlen=100)
        self._mean_q_window: deque = deque(maxlen=100)
        self._td_error_window: deque = deque(maxlen=100)

        self._pinned_obs: Optional[dict] = None
        self._pinned_next_obs: Optional[dict] = None

        self.profile_every: int = 0
        self._profiler = StepProfiler()
        self._rollout_iter = 0

    def _sample_masked_actions(self, action_masks: np.ndarray) -> np.ndarray:
        """Sample actions only from valid mask choices."""
        action_masks = np.asarray(action_masks)
        actions = np.empty(action_masks.shape[0], dtype=np.int64)
        for i, mask in enumerate(action_masks):
            valid = np.flatnonzero(mask)
            actions[i] = np.random.choice(valid) if len(valid) else np.random.randint(self.n_actions)
        return actions

    def _predict_masked(self, observation: dict, action_masks: np.ndarray) -> np.ndarray:
        """Pick the highest-Q action among the valid masks."""
        self.q_net.eval()
        with torch.no_grad():
            obs_t = _obs_to_tensors(observation, self.device)
            q_values = self.q_net(obs_t).cpu().numpy()
        q_values = np.where(np.asarray(action_masks).astype(bool), q_values, -np.inf)
        return q_values.argmax(axis=1)

    def predict(
        self,
        observation: dict,
        state=None,
        episode_start=None,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[object]]:
        """Compute an action for a single observation or batch."""
        vectorized = _is_vectorized(observation, self.observation_space)
        obs_batch = observation if vectorized else {
            "grid_tensor": observation["grid_tensor"][None, ...],
            "features": observation["features"][None, ...],
        }

        if action_masks is None:
            masks = np.ones((obs_batch["features"].shape[0], self.n_actions), dtype=bool)
        else:
            masks = np.asarray(action_masks)
            if masks.ndim == 1:
                masks = masks[None, :]

        if deterministic:
            action = self._predict_masked(obs_batch, masks)
        else:
            explore = np.random.rand(masks.shape[0]) < self.exploration_rate
            if explore.all():
                action = self._sample_masked_actions(masks)
            else:
                action = self._predict_masked(obs_batch, masks)
                if explore.any():
                    action[explore] = self._sample_masked_actions(masks[explore])

        if not vectorized:
            action = action[0]
        return action, state

    def _ensure_replay_buffer(self, n_envs: int) -> None:
        """Create or resize the replay buffer to match the active env count."""
        if self.replay_buffer is None or self.n_envs != n_envs:
            self.n_envs = n_envs
            self.replay_buffer = DictReplayBuffer(
                self.buffer_size, n_envs, self.observation_space, self.n_actions,
            )

    def save_replay_buffer(self, path, max_transitions: Optional[int] = None) -> int:
        """
        Persist the newest `max_transitions` transitions. 
        Returns how many were written.
        """
        if self.replay_buffer is None:
            return 0
        max_rows = None if max_transitions is None else max(1, int(max_transitions) // self.n_envs)
        return self.replay_buffer.save(path, max_rows=max_rows, num_timesteps=self.num_timesteps) * self.n_envs

    def load_replay_buffer(self, path) -> int:
        """
        Restore transitions saved by `save_replay_buffer`. 
        Returns how many were restored.
        """
        self._ensure_replay_buffer(self.n_envs)
        return self.replay_buffer.load(path) * self.n_envs

    def _ensure_pinned_buffers(self, batch_size: int) -> None:
        """Lazily allocate persistent pinned CPU staging buffers."""
        if self.device.type != "cuda":
            return
        grid_shape = tuple(self.observation_space["grid_tensor"].shape)
        feat_shape = tuple(self.observation_space["features"].shape)

        need_alloc = (
            self._pinned_obs is None
            or self._pinned_obs["grid"].shape[0] != batch_size
        )
        if need_alloc:
            def make(shape):
                return torch.empty((batch_size, *shape), dtype=torch.float32, pin_memory=True)

            self._pinned_obs = {"grid": make(grid_shape), "features": make(feat_shape)}
            self._pinned_next_obs = {"grid": make(grid_shape), "features": make(feat_shape)}

    def train_step(self) -> Optional[dict]:
        """Run one gradient step on a minibatch from the replay buffer."""
        if self.replay_buffer is None or len(self.replay_buffer) < max(self.batch_size, 1):
            return None
        if self.replay_buffer.n_sampleable_rows() == 0:
            return None

        do_profile = self.profile_every > 0 and (self.n_updates % self.profile_every == 0)
        prof = self._profiler if do_profile else None
        dev = self.device if do_profile else None

        self._ensure_pinned_buffers(self.batch_size)

        if prof: prof.start("sample")
        batch = self.replay_buffer.sample(self.batch_size)
        if prof: prof.stop("sample")

        if prof: prof.start("host_to_device")
        obs = _obs_to_tensors(batch["obs"], self.device, pinned=self._pinned_obs)
        next_obs = _obs_to_tensors(batch["next_obs"], self.device, pinned=self._pinned_next_obs)
        actions = torch.as_tensor(batch["actions"], device=self.device, dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32)
        dones = torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32)
        next_masks = torch.as_tensor(batch["next_action_masks"], device=self.device, dtype=torch.bool)
        if prof: prof.stop("host_to_device", device=dev)

        if self.symmetry_augmentation:
            if prof: prof.start("augment")
            self._augmenter.augment(obs, next_obs, actions, next_masks)
            if prof: prof.stop("augment", device=dev)

        if prof: prof.start("target_forward")
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self._amp_enabled):
            next_q = self.q_net_target(next_obs)
            next_q_max = next_q.masked_fill(~next_masks, float("-inf")).max(dim=1).values
            next_q_max = torch.where(next_masks.any(dim=1), next_q_max, torch.zeros_like(next_q_max))
            target = rewards + (1.0 - dones) * self.gamma * next_q_max
        if prof: prof.stop("target_forward", device=dev)

        if prof: prof.start("online_forward_backward")
        self.q_net.train()
        with torch.autocast(device_type=self.device.type, enabled=self._amp_enabled):
            q_values_all = self.q_net(obs)
            q_values = q_values_all.gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_at = self.max_grad_norm if (self.max_grad_norm is not None and self.max_grad_norm > 0) else float("inf")
        grad_norm = nn.utils.clip_grad_norm_(self.q_net.parameters(), clip_at)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if prof: prof.stop("online_forward_backward", device=dev)

        with torch.no_grad():
            td_error = (target - q_values).abs().mean()
            mean_q = q_values_all.mean()

        if prof:
            self._last_profile = prof.summary()
            self._last_profile["replay_sample_reported_ms"] = (
                (self.replay_buffer.last_sample_time_s or 0.0) * 1000.0
            )

        return {
            "loss": loss.detach(),
            "grad_norm": grad_norm.detach() if torch.is_tensor(grad_norm) else torch.tensor(float(grad_norm), device=self.device),
            "mean_q": mean_q.detach(),
            "td_error": td_error.detach(),
        }

    def get_profile_stats(self) -> Optional[dict]:
        """Return the most recent profiled train-step breakdown in milliseconds, or None if unavailable."""
        return getattr(self, "_last_profile", None)

    def update_target(self) -> None:
        """Blend the target network toward the online network."""
        with torch.no_grad():
            for target_param, param in zip(self.q_net_target.parameters(), self.q_net.parameters()):
                target_param.data.mul_(1.0 - self.tau).add_(self.tau * param.data)

    def refresh_metrics(self) -> None:
        """Materialize rolling metric windows into their cached mean values with a single CPU sync."""
        if not self._loss_window:
            return
        means = torch.stack([
            torch.stack(list(self._loss_window)).mean(),
            torch.stack(list(self._grad_norm_window)).mean(),
            torch.stack(list(self._mean_q_window)).mean(),
            torch.stack(list(self._td_error_window)).mean(),
        ])
        loss_m, grad_m, q_m, td_m = means.tolist()  # single sync
        self.last_loss_mean = loss_m
        self.last_grad_norm = grad_m
        self.last_mean_q = q_m
        self.last_td_error = td_m

    def learn(self, env, total_timesteps: int, callbacks: Optional[list] = None) -> None:
        """Collect experience and train for the next training chunk."""
        self._ensure_replay_buffer(env.num_envs)
        callbacks = callbacks or []
        target_num_timesteps = self.num_timesteps + int(total_timesteps)

        if self._last_obs is None:
            self._last_obs = env.reset()
            self._last_action_masks = env.action_masks()

        for cb in callbacks:
            cb.on_training_start(self, env)

        while self.num_timesteps < target_num_timesteps:
            do_profile = self.profile_every > 0 and (self._rollout_iter % self.profile_every == 0)
            prof = self._profiler if do_profile else None
            dev = self.device if do_profile else None

            for cb in callbacks:
                cb.on_rollout_start(self, env)

            self.exploration_rate = self.exploration_schedule.value(self.num_timesteps)

            if prof: prof.start("action_select")
            if self.num_timesteps < self.learning_starts:
                actions = self._sample_masked_actions(self._last_action_masks)
            else:
                actions, _ = self.predict(
                    self._last_obs, deterministic=False, action_masks=self._last_action_masks,
                )
            if prof: prof.stop("action_select", device=dev)

            if prof: prof.start("env_step")
            next_obs, rewards, dones, infos = env.step(actions)
            if prof: prof.stop("env_step")

            if prof: prof.start("action_masks")
            next_action_masks = env.action_masks()
            if prof: prof.stop("action_masks")

            if prof: prof.start("buffer_add")
            self.replay_buffer.add(
                self._last_obs, next_obs, actions, rewards, dones,
                self._last_action_masks, next_action_masks,
            )
            if prof: prof.stop("buffer_add")

            self._rollout_iter += 1

            prev_num_timesteps = self.num_timesteps
            self._last_obs = next_obs
            self._last_action_masks = next_action_masks
            self.num_timesteps += self.n_envs
            self._steps_since_train += 1

            if prof: prof.start("callbacks_on_step")
            for cb in callbacks:
                cb.on_step(self, env)
            if prof: prof.stop("callbacks_on_step")

            buffer_ready = len(self.replay_buffer) >= min(self.learning_starts, self.replay_buffer.capacity)
            if not buffer_ready:
                self._steps_since_train = 0
            elif self._steps_since_train >= self.train_freq:
                n_updates = self.gradient_steps if self.gradient_steps > 0 else self._steps_since_train
                for _ in range(max(1, n_updates)):
                    metrics = self.train_step()
                    if metrics is not None:
                        self._loss_window.append(metrics["loss"])
                        self._grad_norm_window.append(metrics["grad_norm"])
                        self._mean_q_window.append(metrics["mean_q"])
                        self._td_error_window.append(metrics["td_error"])
                        self.n_updates += 1
                self._steps_since_train = 0

            crossed = (
                self.num_timesteps // self.target_update_interval
                > prev_num_timesteps // self.target_update_interval
            )
            if crossed:
                self.update_target()

    def _hyperparams(self) -> dict:
        """Return the optimizer and training hyperparameters."""
        return dict(
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            tau=self.tau,
            gamma=self.gamma,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            target_update_interval=self.target_update_interval,
            exploration_fraction=self.exploration_fraction,
            exploration_initial_eps=self.exploration_initial_eps,
            exploration_final_eps=self.exploration_final_eps,
            max_grad_norm=self.max_grad_norm,
            features_dim=self.features_dim,
            exploration_duration=self.exploration_duration,
            symmetry_augmentation=self.symmetry_augmentation,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
        )

    def save(self, path) -> None:
        """Save the model checkpoint to disk."""
        checkpoint = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "hyperparams": self._hyperparams(),
            "num_timesteps": self.num_timesteps,
            "exploration_rate": self.exploration_rate,
            "n_updates": self.n_updates,
            "q_net_state_dict": self.q_net.state_dict(),
            "q_net_target_state_dict": self.q_net_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, env=None, device: str = "cpu") -> "MaskableDQN":
        """Load a model from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        n_envs = env.num_envs if env is not None else checkpoint.get("n_envs", 1)

        model = cls(
            checkpoint["observation_space"],
            checkpoint["action_space"],
            n_envs=n_envs,
            device=device,
            **checkpoint["hyperparams"],
        )
        model.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        model.q_net_target.load_state_dict(checkpoint["q_net_target_state_dict"])

        if env is not None and checkpoint.get("optimizer_state_dict") is not None:
            try:
                model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except ValueError:
                pass
        if checkpoint.get("scaler_state_dict") is not None:
            try:
                model.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except (ValueError, RuntimeError):
                pass

        model.num_timesteps = checkpoint.get("num_timesteps", 0)
        model.exploration_rate = checkpoint.get("exploration_rate", model.exploration_initial_eps)
        model.n_updates = checkpoint.get("n_updates", 0)
        return model