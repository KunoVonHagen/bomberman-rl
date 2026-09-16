from __future__ import annotations

import numpy as np
import torch as th
from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableDictRolloutBufferSamples


class PinnedMaskableDictRolloutBuffer(MaskableDictRolloutBuffer):
    def reset(self) -> None:
        super().reset()
        self._obs_views: dict = {}
        self._staging_slot = 0
        if not hasattr(self, "_staging"):
            self._staging: list = [{}, {}]
            self._staging_events: list = [None, None]

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
        return MaskableDictRolloutBufferSamples(
            observations=self._gather_observations(batch_inds),
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
            action_masks=self.to_torch(self.action_masks[batch_inds].reshape(-1, self.mask_dims)),
        )
