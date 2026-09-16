from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableDictRolloutBufferSamples

_STEP_FIELDS = ("rewards", "returns", "episode_starts", "values", "log_probs", "advantages")
_FLAT_FIELDS = ("actions", "values", "log_probs", "advantages", "returns", "action_masks")


class PinnedMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    def reset(self) -> None:
        if getattr(self, "_arrays_ready", False):
            for key, obs in self.observations.items():
                self.observations[key] = obs.reshape((self.buffer_size, self.n_envs) + tuple(self.obs_shape[key]))
            self.actions = self.actions.reshape(self.buffer_size, self.n_envs, self.action_dim)
            for name in _STEP_FIELDS:
                setattr(self, name, getattr(self, name).reshape(self.buffer_size, self.n_envs))
            self.action_masks = self.action_masks.reshape(self.buffer_size, self.n_envs, self.mask_dims)
            self.pos = 0
            self.full = False
            self.generator_ready = False
        else:
            super().reset()
            self._arrays_ready = True
        self._obs_views: dict = {}
        self._staging_slot = 0
        if not hasattr(self, "_staging"):
            self._staging: list = [{}, {}]
            self._staging_events: list = [None, None]

    def add(self, obs: dict, action: np.ndarray, reward: np.ndarray, episode_start: np.ndarray,
            value: th.Tensor, log_prob: th.Tensor, action_masks=None) -> None:
        if action_masks is not None:
            self.action_masks[self.pos] = np.reshape(action_masks, (self.n_envs, self.mask_dims))
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)
        for key, target in self.observations.items():
            target[self.pos] = obs[key]
        self.actions[self.pos] = np.reshape(action, (self.n_envs, self.action_dim))
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size=None):
        assert self.full, ""
        total = self.buffer_size * self.n_envs
        indices = np.random.permutation(total)
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = obs.reshape((total,) + tuple(obs.shape[2:]))
            for name in _FLAT_FIELDS:
                arr = self.__dict__[name]
                shape = arr.shape if arr.ndim >= 3 else (*arr.shape, 1)
                self.__dict__[name] = arr.reshape((shape[0] * shape[1], *shape[2:]))
            self.generator_ready = True
        if batch_size is None:
            batch_size = total
        start_idx = 0
        while start_idx < total:
            yield self._get_samples(indices[start_idx:start_idx + batch_size])
            start_idx += batch_size

    def _obs_view(self, key: str) -> th.Tensor:
        array = self.observations[key]
        view = self._obs_views.get(key)
        if view is None or view.data_ptr() != array.ctypes.data or tuple(view.shape) != array.shape:
            view = th.from_numpy(array)
            self._obs_views[key] = view
        return view

    def _staging_tensor(self, slot: int, key: str, n: int, view: th.Tensor) -> th.Tensor:
        staging = self._staging[slot].get(key)
        shape = (n,) + tuple(view.shape[1:])
        if staging is None or staging.dtype != view.dtype or tuple(staging.shape[1:]) != shape[1:] or staging.shape[0] < n:
            staging = th.empty(shape, dtype=view.dtype, pin_memory=self.device.type == "cuda")
            self._staging[slot][key] = staging
        return staging if staging.shape[0] == n else staging[:n]

    def _gather_observations(self, batch_inds: np.ndarray) -> dict:
        idx = th.from_numpy(np.ascontiguousarray(batch_inds, dtype=np.int64))
        slot = self._staging_slot
        self._staging_slot ^= 1
        event = self._staging_events[slot]
        if event is not None:
            event.synchronize()
        cuda = self.device.type == "cuda"
        out = {}
        for key in self.observations:
            view = self._obs_view(key)
            staging = self._staging_tensor(slot, key, idx.shape[0], view)
            th.index_select(view, 0, idx, out=staging)
            out[key] = staging.to(self.device, non_blocking=cuda)
        if cuda:
            event = th.cuda.Event()
            event.record(th.cuda.current_stream(self.device))
            self._staging_events[slot] = event
        return out

    def _get_samples(self, batch_inds: np.ndarray, env=None) -> MaskableDictRolloutBufferSamples:
        batch_inds = (batch_inds % self.buffer_size) * self.n_envs + batch_inds // self.buffer_size
        return MaskableDictRolloutBufferSamples(
            observations=self._gather_observations(batch_inds),
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )
