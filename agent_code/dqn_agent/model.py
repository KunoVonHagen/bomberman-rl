from __future__ import annotations

import time
import contextlib
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from .config import EXPLORATION_MODES
from .replay_buffer import GRID_SENTINEL_CODE, DeviceReplayBuffer, DictReplayBuffer
from .schedules import GeometricSchedule, LinearSchedule
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
        self.add(name, (time.perf_counter() - self._t0[name]) * 1000.0)

    def add(self, name: str, dt_ms: float) -> None:
        bucket = self._times.setdefault(name, deque(maxlen=self.window))
        bucket.append(dt_ms)

    def summary(self) -> dict[str, float]:
        """Mean milliseconds per stage over the rolling window."""
        return {name: (sum(vals) / len(vals)) for name, vals in self._times.items() if vals}


AMP_MODES = ("off", "fp16")
REPLAY_DEVICES = ("host", "device")


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
        grid_tensor = observations["grid_tensor"]
        if grid_tensor.dtype != torch.float32 and not torch.is_autocast_enabled(grid_tensor.device.type):
            grid_tensor = grid_tensor.float()
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


def _decode_grid(codes: torch.Tensor, codec: tuple, dtype: torch.dtype) -> torch.Tensor:
    den, sentinel = codec
    grid = codes.to(torch.float32).div_(den)
    grid.masked_fill_((codes == GRID_SENTINEL_CODE) & sentinel, -1.0)
    return grid if dtype == torch.float32 else grid.to(dtype)


def _obs_to_tensors(observation: dict, device: torch.device, pinned: Optional[dict] = None,
                    codec: Optional[tuple] = None, dtype: torch.dtype = torch.float32) -> dict:
    """Convert observation arrays to PyTorch tensors on a device."""
    grid = observation["grid_tensor"]
    grid_t = grid if torch.is_tensor(grid) else torch.from_numpy(np.ascontiguousarray(grid))
    feats = observation["features"]
    feats_t = feats.to(torch.float32) if torch.is_tensor(feats) else torch.as_tensor(np.asarray(feats), dtype=torch.float32)
    non_blocking = device.type == "cuda"

    if non_blocking and pinned is not None and feats_t.device.type == "cpu":
        if pinned["grid"].dtype == grid_t.dtype and pinned["grid"].data_ptr() != grid_t.data_ptr():
            pinned["grid"].copy_(grid_t)
            grid_t = pinned["grid"]
        pinned["features"].copy_(feats_t)
        feats_t = pinned["features"]

    grid_d = grid_t.to(device, non_blocking=non_blocking)
    if codec is not None and grid_d.dtype == torch.uint8:
        grid_d = _decode_grid(grid_d, codec, dtype)
    else:
        grid_d = grid_d.to(dtype)
    return {
        "grid_tensor": grid_d,
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
        exploration_mode: str = "epsilon",
        softmax_beta_initial: float = 1.0,
        softmax_beta_final: float = 20.0,
        max_grad_norm: float = 10.0,
        features_dim: int = 256,
        exploration_duration: int = 300_000,
        symmetry_augmentation: bool = False,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        amp: str = "fp16",
        grid_codec=None,
        replay_prefetch: bool = True,
        replay_device: str = "host",
        n_step: int = 1,
        priority_alpha: float = 0.0,
        priority_eps: float = 0.01,
        n_envs: int = 1,
        device: str = "cpu",
        inference: bool = False,
    ):
        self.inference = bool(inference)
        self.n_step = max(1, int(n_step))
        self.priority_alpha = max(0.0, float(priority_alpha))
        self.priority_eps = max(0.0, float(priority_eps))
        self._pending_priorities: deque = deque()
        self._priority_staging: Optional[list] = None
        self._priority_stage_next = 0
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
        if amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {amp!r}")
        self.amp = amp
        self.grid_codec = None
        self._codec_tensors = None
        if grid_codec is not None:
            den, sentinel = grid_codec
            self.grid_codec = (np.ascontiguousarray(den, dtype=np.float32), np.ascontiguousarray(sentinel, dtype=np.bool_))
            self._codec_tensors = (
                torch.as_tensor(self.grid_codec[0], device=self.device).view(-1, 1, 1),
                torch.as_tensor(self.grid_codec[1], device=self.device).view(-1, 1, 1),
            )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.exploration_duration = max(1, int(exploration_duration))
        self.exploration_schedule = LinearSchedule(
            exploration_initial_eps, exploration_final_eps, self.exploration_duration,
        )
        self.exploration_rate = exploration_initial_eps
        if exploration_mode not in EXPLORATION_MODES:
            raise ValueError(f"exploration_mode must be one of {EXPLORATION_MODES}, got {exploration_mode!r}")
        self.exploration_mode = exploration_mode
        self.softmax_beta_initial = float(softmax_beta_initial)
        self.softmax_beta_final = float(softmax_beta_final)
        self.softmax_beta_schedule = GeometricSchedule(
            self.softmax_beta_initial, self.softmax_beta_final, self.exploration_duration,
        )
        self.softmax_beta = self.softmax_beta_initial

        self.q_net = QNetwork(observation_space, self.n_actions, features_dim, self.dropout).to(self.device)
        self.q_net.eval()
        self._amp_enabled = self.device.type == "cuda" and self.amp == "fp16"
        self.replay_prefetch = bool(replay_prefetch)
        if replay_device not in REPLAY_DEVICES:
            raise ValueError(f"replay_device must be one of {REPLAY_DEVICES}, got {replay_device!r}")
        self.replay_device = replay_device
        self._pinned_pairs: Optional[list] = None
        self._pair_events: list = [None, None]
        self._next_pair = 0
        self._side_stream = None
        self._prefetched: Optional[dict] = None
        self._prefetched_from = None
        if self.inference:
            self.q_net_target = None
            self.optimizer = None
            self.scaler = None
        else:
            self.q_net_target = QNetwork(observation_space, self.n_actions, features_dim, self.dropout).to(self.device)
            self.q_net_target.load_state_dict(self.q_net.state_dict())
            self.q_net_target.eval()
            self.optimizer = torch.optim.Adam(
                self.q_net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay,
                fused=self.device.type == "cuda",
            )
            self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

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


        self.profile_every: int = 0
        self._predict_staging: Optional[dict] = None
        self._profiler = StepProfiler()
        self._wall_profiler = StepProfiler(window=200)
        self._rollout_iter = 0

    def _sample_masked_actions(self, action_masks: np.ndarray) -> np.ndarray:
        """Sample actions only from valid mask choices."""
        action_masks = np.asarray(action_masks, dtype=bool)
        if action_masks.ndim == 1:
            action_masks = action_masks[None, :]
        actions = (np.random.rand(*action_masks.shape) * action_masks).argmax(axis=1)
        empty = ~action_masks.any(axis=1)
        if empty.any():
            actions[empty] = np.random.randint(self.n_actions, size=int(empty.sum()))
        return actions

    def _sample_softmax_actions(self, q_values: np.ndarray, action_masks: np.ndarray, beta: float) -> np.ndarray:
        masks = np.asarray(action_masks, dtype=bool)
        if masks.ndim == 1:
            masks = masks[None, :]
        valid = masks.any(axis=1)
        z = np.where(masks, float(beta) * np.asarray(q_values, dtype=np.float64), -np.inf)
        z[~valid] = 0.0
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        cdf = np.cumsum(p, axis=1)
        u = np.random.rand(masks.shape[0], 1) * cdf[:, -1:]
        hit = cdf > u
        actions = hit.argmax(axis=1)
        none = ~hit.any(axis=1)
        if none.any():
            actions[none] = masks.shape[1] - 1 - masks[none, ::-1].argmax(axis=1)
        if (~valid).any():
            actions[~valid] = np.random.randint(self.n_actions, size=int((~valid).sum()))
        return actions

    def _predict_tensors(self, observation: dict) -> dict:
        if self.device.type != "cuda":
            return _obs_to_tensors(observation, self.device)
        grid = observation["grid_tensor"]
        grid_t = grid if torch.is_tensor(grid) else torch.from_numpy(np.ascontiguousarray(grid))
        feats_t = torch.as_tensor(np.asarray(observation["features"]), dtype=torch.float32)
        n = grid_t.shape[0]
        staging = self._predict_staging
        if staging is None or staging["grid"].shape[0] < n or staging["grid"].shape[1:] != grid_t.shape[1:]:
            staging = self._predict_staging = {
                "grid": torch.empty(tuple(grid_t.shape), dtype=torch.float16, pin_memory=True),
                "features": torch.empty(tuple(feats_t.shape), dtype=torch.float32, pin_memory=True),
            }
        staging["grid"][:n].copy_(grid_t)
        staging["features"][:n].copy_(feats_t)
        return {
            "grid_tensor": staging["grid"][:n].to(self.device, non_blocking=True),
            "features": staging["features"][:n].to(self.device, non_blocking=True),
        }

    def q_values(self, observation: dict, mc_dropout_samples: int = 0) -> np.ndarray:
        samples = int(mc_dropout_samples) if self.dropout > 0 else 0
        if self.q_net.training != (samples > 0):
            self.q_net.train(samples > 0)
        with torch.no_grad():
            obs_t = self._predict_tensors(observation)
            if samples > 0:
                batch = obs_t["features"].shape[0]
                tiled = {key: value.repeat(samples, *([1] * (value.dim() - 1))) for key, value in obs_t.items()}
                q = self.q_net(tiled).view(samples, batch, -1).mean(dim=0)
                self.q_net.eval()
            else:
                q = self.q_net(obs_t)
        return q.cpu().numpy()

    def _predict_masked(self, observation: dict, action_masks: np.ndarray) -> np.ndarray:
        """Pick the highest-Q action among the valid masks."""
        q_values = self.q_values(observation)
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
        elif self.exploration_mode == "softmax":
            action = self._sample_softmax_actions(self.q_values(obs_batch), masks, self.softmax_beta)
            explore = np.random.rand(masks.shape[0]) < self.exploration_final_eps
            if explore.any():
                action[explore] = self._sample_masked_actions(masks[explore])
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
            self._prefetched = None
            self._pending_priorities.clear()
            if self.replay_device == "device":
                self.replay_buffer = DeviceReplayBuffer(
                    self.buffer_size, n_envs, self.observation_space, self.n_actions, self.device,
                    grid_codec=self.grid_codec, priority_alpha=self.priority_alpha, priority_eps=self.priority_eps,
                    priority_n_step=self.n_step,
                )
            else:
                self.replay_buffer = DictReplayBuffer(
                    self.buffer_size, n_envs, self.observation_space, self.n_actions, grid_codec=self.grid_codec,
                    priority_alpha=self.priority_alpha, priority_eps=self.priority_eps, priority_n_step=self.n_step,
                )

    def set_priority_options(self, alpha: float, eps: float) -> None:
        self.priority_alpha = max(0.0, float(alpha))
        self.priority_eps = max(0.0, float(eps))
        if self.replay_buffer is not None:
            self.replay_buffer.set_priority_alpha(self.priority_alpha, self.priority_eps, self.n_step)

    def _flush_priority_updates(self, force: bool = False) -> None:
        while self._pending_priorities:
            indices, versions, td_abs, event = self._pending_priorities[0]
            if event is not None:
                if not (force or event.query()):
                    break
                event.synchronize()
                force = False
            self._pending_priorities.popleft()
            self.replay_buffer.update_priorities(indices, versions, td_abs.numpy())

    def _queue_priority_update(self, indices: np.ndarray, versions: np.ndarray, td_abs: torch.Tensor) -> None:
        if self.device.type != "cuda":
            self.replay_buffer.update_priorities(indices, versions, td_abs.cpu().numpy())
            return
        n_slots = 4
        if self._priority_staging is None or self._priority_staging[0].shape[0] != td_abs.shape[0]:
            self._priority_staging = [torch.empty(td_abs.shape[0], dtype=torch.float32, pin_memory=True) for _ in range(n_slots)]
            self._priority_stage_next = 0
        if len(self._pending_priorities) >= n_slots:
            self._flush_priority_updates(force=True)
        host = self._priority_staging[self._priority_stage_next]
        self._priority_stage_next = (self._priority_stage_next + 1) % n_slots
        host.copy_(td_abs, non_blocking=True)
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self._pending_priorities.append((indices, versions, host, event))

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
        if self.device.type != "cuda" or isinstance(self.replay_buffer, DeviceReplayBuffer):
            return
        n_pairs = 2 if self.replay_prefetch else 1
        if self._pinned_pairs is not None and len(self._pinned_pairs) == n_pairs \
                and self._pinned_pairs[0][0]["grid"].shape[0] == batch_size:
            return
        grid_shape = tuple(self.observation_space["grid_tensor"].shape)
        feat_shape = tuple(self.observation_space["features"].shape)
        grid_dtype = torch.float16 if self.replay_buffer is None else torch.from_numpy(self.replay_buffer.grid[:1, :1]).dtype

        def make(shape, dtype):
            return torch.empty((batch_size, *shape), dtype=dtype, pin_memory=True)

        self._pinned_pairs = [
            ({"grid": make(grid_shape, grid_dtype), "features": make(feat_shape, torch.float32)},
             {"grid": make(grid_shape, grid_dtype), "features": make(feat_shape, torch.float32)})
            for _ in range(n_pairs)
        ]
        self._pair_events = [None] * n_pairs
        self._next_pair = 0
        self._prefetched = None

    def _fetch_batch(self, pair: int, stream=None) -> dict:
        cuda = self.device.type == "cuda"
        pinned_obs = pinned_next = None
        staging = None
        if self._pinned_pairs is not None:
            pinned_obs, pinned_next = self._pinned_pairs[pair]
            event = self._pair_events[pair]
            if event is not None:
                event.synchronize()
            staging = {"grid": pinned_obs["grid"], "next_grid": pinned_next["grid"]}
        train_dtype = torch.float16 if self._amp_enabled else torch.float32
        context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with context:
            self._flush_priority_updates()
            batch = self.replay_buffer.sample(self.batch_size, out=staging, n_step=self.n_step, gamma=self.gamma)
            fetched = dict(
                obs=_obs_to_tensors(batch["obs"], self.device, pinned=pinned_obs, codec=self._codec_tensors, dtype=train_dtype),
                next_obs=_obs_to_tensors(batch["next_obs"], self.device, pinned=pinned_next, codec=self._codec_tensors, dtype=train_dtype),
                actions=torch.as_tensor(batch["actions"], device=self.device, dtype=torch.int64),
                rewards=torch.as_tensor(batch["rewards"], device=self.device, dtype=torch.float32),
                dones=torch.as_tensor(batch["dones"], device=self.device, dtype=torch.float32),
                discounts=torch.as_tensor(batch["discounts"], device=self.device, dtype=torch.float32),
                next_masks=torch.as_tensor(batch["next_action_masks"], device=self.device, dtype=torch.bool),
                indices=batch["indices"],
                versions=batch["versions"],
            )
            if cuda:
                event = torch.cuda.Event()
                event.record(stream if stream is not None else torch.cuda.current_stream(self.device))
                if self._pinned_pairs is not None:
                    self._pair_events[pair] = event
                fetched["event"] = event
        return fetched

    def _prefetch_next_batch(self) -> None:
        if self.replay_buffer.n_sampleable_rows() == 0:
            return
        if self._side_stream is None:
            self._side_stream = torch.cuda.Stream(self.device)
        pair = self._next_pair
        self._next_pair = (pair + 1) % (len(self._pinned_pairs) if self._pinned_pairs is not None else 2)
        self._prefetched = self._fetch_batch(pair, stream=self._side_stream)
        self._prefetched_from = self.replay_buffer
        torch.cuda.current_stream(self.device).wait_event(self._prefetched["event"])

    def _take_batch(self) -> dict:
        fetched = self._prefetched
        self._prefetched = None
        if fetched is not None and self._prefetched_from is self.replay_buffer:
            current = torch.cuda.current_stream(self.device)
            current.wait_event(fetched["event"])
            for value in fetched.values():
                if torch.is_tensor(value):
                    value.record_stream(current)
                elif isinstance(value, dict):
                    for tensor in value.values():
                        tensor.record_stream(current)
            return fetched
        pair = self._next_pair
        self._next_pair = (pair + 1) % (len(self._pinned_pairs) if self._pinned_pairs is not None else 2)
        return self._fetch_batch(pair)

    def train_step(self) -> Optional[dict]:
        """Run one gradient step on a minibatch from the replay buffer."""
        if self.replay_buffer is None or len(self.replay_buffer) < max(self.batch_size, 1):
            return None
        if self.replay_buffer.n_sampleable_rows() == 0:
            return None

        do_profile = self.profile_every > 0 and (self.n_updates % self.profile_every == self.profile_every // 2)
        prof = self._profiler if do_profile else None
        dev = self.device if do_profile else None

        self._ensure_pinned_buffers(self.batch_size)

        if prof: prof.start("sample")
        fetched = self._take_batch()
        obs, next_obs = fetched["obs"], fetched["next_obs"]
        actions, rewards, dones, next_masks = fetched["actions"], fetched["rewards"], fetched["dones"], fetched["next_masks"]
        discounts = fetched["discounts"]
        if prof: prof.stop("sample", device=dev)

        if self.symmetry_augmentation:
            if prof: prof.start("augment")
            self._augmenter.augment(obs, next_obs, actions, next_masks)
            if prof: prof.stop("augment", device=dev)

        if prof: prof.start("target_forward")
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self._amp_enabled):
            next_q = self.q_net_target(next_obs)
            next_q_max = next_q.masked_fill(~next_masks, float("-inf")).max(dim=1).values
            next_q_max = torch.where(next_masks.any(dim=1), next_q_max, torch.zeros_like(next_q_max))
            target = rewards + (1.0 - dones) * discounts * next_q_max
        if prof: prof.stop("target_forward", device=dev)

        if prof: prof.start("online_forward_backward")
        if self.dropout > 0:
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
        if self.dropout > 0:
            self.q_net.eval()
        if prof: prof.stop("online_forward_backward", device=dev)

        if self.replay_prefetch and self.device.type == "cuda":
            if prof: prof.start("prefetch")
            self._prefetch_next_batch()
            if prof: prof.stop("prefetch")

        with torch.no_grad():
            td_abs = (target - q_values).abs()
            td_error = td_abs.mean()
            mean_q = q_values_all.mean()
            if self.replay_buffer.priority_alpha > 0.0:
                self._queue_priority_update(fetched["indices"], fetched["versions"], td_abs.float())

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
        stats = dict(getattr(self, "_last_profile", None) or {})
        if self.profile_every > 0:
            stats.update({f"wall_{name}": ms for name, ms in self._wall_profiler.summary().items()})
        return stats or None

    def update_target(self) -> None:
        """Blend the target network toward the online network."""
        with torch.no_grad():
            targets = [p.data for p in self.q_net_target.parameters()]
            sources = [p.data for p in self.q_net.parameters()]
            if self.tau >= 1.0:
                torch._foreach_copy_(targets, sources)
            else:
                torch._foreach_mul_(targets, 1.0 - self.tau)
                torch._foreach_add_(targets, sources, alpha=self.tau)

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
        if self.inference:
            raise RuntimeError("this MaskableDQN was loaded with inference=True and cannot be trained")
        self._ensure_replay_buffer(env.num_envs)
        callbacks = callbacks or []
        target_num_timesteps = self.num_timesteps + int(total_timesteps)

        if self._last_obs is None:
            self._last_obs = env.reset()
            self._last_action_masks = env.action_masks()

        for cb in callbacks:
            cb.on_training_start(self, env)

        while self.num_timesteps < target_num_timesteps:
            do_profile = self.profile_every > 0 and (self._rollout_iter % self.profile_every == self.profile_every // 2)
            prof = self._profiler if do_profile else None
            dev = self.device if do_profile else None
            wall = self._wall_profiler if self.profile_every > 0 else None
            if wall: wall.start("iteration")

            for cb in callbacks:
                cb.on_rollout_start(self, env)

            self.exploration_rate = self.exploration_schedule.value(self.num_timesteps)
            self.softmax_beta = self.softmax_beta_schedule.value(self.num_timesteps)

            if prof: prof.start("action_select")
            if wall: wall.start("action_select")
            if self.num_timesteps < self.learning_starts:
                actions = self._sample_masked_actions(self._last_action_masks)
            else:
                actions, _ = self.predict(
                    self._last_obs, deterministic=False, action_masks=self._last_action_masks,
                )
            if wall: wall.stop("action_select")
            if prof: prof.stop("action_select", device=dev)

            if prof: prof.start("env_step")
            env.step_async(actions)
            if prof: prof.stop("env_step")
            if wall: wall.start("train")

            buffer_ready = len(self.replay_buffer) >= min(self.learning_starts, self.replay_buffer.capacity)
            if not buffer_ready:
                self._steps_since_train = 0
            elif self._steps_since_train >= self.train_freq:
                if prof: prof.start("train")
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
                if prof: prof.stop("train", device=dev)
            if wall: wall.stop("train")

            if prof: prof.start("env_wait")
            if wall: wall.start("env_wait")
            next_obs, rewards, dones, infos = env.step_wait()
            if wall:
                wall.stop("env_wait")
                wall.add("opponent_infer", getattr(getattr(env, "venv", env), "opponent_eval_ms", 0.0))
            if prof: prof.stop("env_wait")

            if prof: prof.start("action_masks")
            next_action_masks = env.action_masks()
            if prof: prof.stop("action_masks")

            if prof: prof.start("buffer_add")
            if wall: wall.start("buffer_add")
            self.replay_buffer.add(
                self._last_obs, next_obs, actions, rewards, dones,
                self._last_action_masks, next_action_masks,
            )
            if wall: wall.stop("buffer_add")
            if prof: prof.stop("buffer_add")

            self._rollout_iter += 1

            prev_num_timesteps = self.num_timesteps
            self._last_obs = next_obs
            self._last_action_masks = next_action_masks
            self.num_timesteps += self.n_envs
            self._steps_since_train += 1

            if prof: prof.start("callbacks_on_step")
            if wall: wall.start("callbacks_on_step")
            for cb in callbacks:
                cb.on_step(self, env)
            if wall: wall.stop("callbacks_on_step")
            if prof: prof.stop("callbacks_on_step")
            if wall: wall.stop("iteration")

            crossed = (
                self.num_timesteps // self.target_update_interval
                > prev_num_timesteps // self.target_update_interval
            )
            if crossed:
                self.update_target()

    def set_replay_options(self, prefetch: bool, replay_device: str) -> None:
        if replay_device not in REPLAY_DEVICES:
            raise ValueError(f"replay_device must be one of {REPLAY_DEVICES}, got {replay_device!r}")
        self.replay_prefetch = bool(prefetch)
        self._prefetched = None
        if replay_device != self.replay_device:
            if self.replay_buffer is not None:
                raise RuntimeError("replay_device cannot change after the replay buffer was created")
            self.replay_device = replay_device

    def set_amp(self, amp: str) -> None:
        if amp not in AMP_MODES:
            raise ValueError(f"amp must be one of {AMP_MODES}, got {amp!r}")
        if amp == self.amp:
            return
        self.amp = amp
        self._amp_enabled = self.device.type == "cuda" and amp == "fp16"
        if self.scaler is not None:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

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
            exploration_mode=self.exploration_mode,
            softmax_beta_initial=self.softmax_beta_initial,
            softmax_beta_final=self.softmax_beta_final,
            max_grad_norm=self.max_grad_norm,
            features_dim=self.features_dim,
            exploration_duration=self.exploration_duration,
            symmetry_augmentation=self.symmetry_augmentation,
            weight_decay=self.weight_decay,
            dropout=self.dropout,
            amp=self.amp,
            grid_codec=self.grid_codec,
            n_step=self.n_step,
            priority_alpha=self.priority_alpha,
            priority_eps=self.priority_eps,
        )

    def save(self, path) -> None:
        """Save the model checkpoint to disk."""
        if self.inference:
            raise RuntimeError("this MaskableDQN was loaded with inference=True and cannot be saved")
        checkpoint = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "hyperparams": self._hyperparams(),
            "num_timesteps": self.num_timesteps,
            "exploration_rate": self.exploration_rate,
            "softmax_beta": self.softmax_beta,
            "n_updates": self.n_updates,
            "q_net_state_dict": self.q_net.state_dict(),
            "q_net_target_state_dict": self.q_net_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path, env=None, device: str = "cpu", inference: bool = False) -> "MaskableDQN":
        """Load a model from a checkpoint file."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        n_envs = env.num_envs if env is not None else checkpoint.get("n_envs", 1)

        model = cls(
            checkpoint["observation_space"],
            checkpoint["action_space"],
            n_envs=n_envs,
            device=device,
            inference=inference,
            **checkpoint["hyperparams"],
        )
        model.q_net.load_state_dict(checkpoint["q_net_state_dict"])
        if inference:
            model.num_timesteps = checkpoint.get("num_timesteps", 0)
            model.exploration_rate = checkpoint.get("exploration_rate", model.exploration_initial_eps)
            model.softmax_beta = checkpoint.get("softmax_beta", model.softmax_beta_final)
            model.n_updates = checkpoint.get("n_updates", 0)
            return model
        model.q_net_target.load_state_dict(checkpoint["q_net_target_state_dict"])

        if env is not None and checkpoint.get("optimizer_state_dict") is not None:
            optimizer_state = checkpoint["optimizer_state_dict"]
            fused = model.optimizer.param_groups[0].get("fused")
            for group in optimizer_state.get("param_groups", []):
                group["fused"] = fused
            try:
                model.optimizer.load_state_dict(optimizer_state)
            except ValueError:
                pass
        if checkpoint.get("scaler_state_dict") is not None:
            try:
                model.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            except (ValueError, RuntimeError):
                pass

        model.num_timesteps = checkpoint.get("num_timesteps", 0)
        model.exploration_rate = checkpoint.get("exploration_rate", model.exploration_initial_eps)
        model.softmax_beta = checkpoint.get("softmax_beta", model.softmax_beta_final)
        model.n_updates = checkpoint.get("n_updates", 0)
        return model