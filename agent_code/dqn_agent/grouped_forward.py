from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

EXTRACTOR_PREFIX = "features_extractor."
GROUP_NORM_GROUPS = 8
GROUP_NORM_EPS = 1e-5


class StackedParameters:
    def __init__(self, state_dicts: Sequence[Dict[str, torch.Tensor]], keys: Sequence[str]):
        self.k = len(state_dicts)
        self.params: Dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [sd[key] for sd in state_dicts]
            shape = tuple(tensors[0].shape)
            if any(tuple(t.shape) != shape for t in tensors):
                raise ValueError(f"parameter {key} differs in shape between the stacked networks")
            self.params[key] = torch.stack([t.detach().to(torch.float32) for t in tensors]).contiguous()

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.params[key]


def stack_state_dicts(state_dicts: Sequence[Dict[str, torch.Tensor]], keys: Sequence[str]) -> Optional[StackedParameters]:
    for sd in state_dicts:
        if any(key not in sd for key in keys):
            return None
    try:
        return StackedParameters(state_dicts, keys)
    except ValueError:
        return None


UNFOLD_ROW_CHUNK = 128


def grouped_conv3x3(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    if x.device.type == "cuda":
        return _grouped_conv3x3_unfold(x, weight, bias, idx)
    n, c_in, h, w = x.shape
    weights = weight.index_select(0, idx)
    c_out = weights.shape[1]
    out = F.conv2d(
        x.reshape(1, n * c_in, h, w),
        weights.reshape(n * c_out, c_in, 3, 3),
        bias.index_select(0, idx).reshape(-1),
        padding=1,
        groups=n,
    )
    return out.reshape(n, c_out, h, w)


def _grouped_conv3x3_unfold(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    n, c_in, h, w = x.shape
    c_out = weight.shape[1]
    out = x.new_empty((n, c_out, h * w))
    for start in range(0, n, UNFOLD_ROW_CHUNK):
        stop = min(n, start + UNFOLD_ROW_CHUNK)
        rows = idx[start:stop]
        cols = F.unfold(x[start:stop], 3, padding=1)
        weights = weight.index_select(0, rows).reshape(stop - start, c_out, c_in * 9)
        biases = bias.index_select(0, rows).unsqueeze(2)
        torch.baddbmm(biases, weights, cols, out=out[start:stop])
    return out.view(n, c_out, h, w)


def grouped_group_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    n, c, h, w = x.shape
    out = F.group_norm(
        x.reshape(1, n * c, h, w),
        n * GROUP_NORM_GROUPS,
        weight.index_select(0, idx).reshape(-1),
        bias.index_select(0, idx).reshape(-1),
        GROUP_NORM_EPS,
    )
    return out.reshape(n, c, h, w)


def grouped_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    weights = weight.index_select(0, idx)
    biases = bias.index_select(0, idx)
    return torch.baddbmm(biases.unsqueeze(1), x.unsqueeze(1), weights.transpose(1, 2)).squeeze(1)


EXTRACTOR_KEYS = [
    "stem.0.weight", "stem.0.bias", "stem.1.weight", "stem.1.bias",
    "res_blocks.0.block.0.weight", "res_blocks.0.block.0.bias", "res_blocks.0.block.1.weight", "res_blocks.0.block.1.bias",
    "res_blocks.0.block.3.weight", "res_blocks.0.block.3.bias", "res_blocks.0.block.4.weight", "res_blocks.0.block.4.bias",
    "res_blocks.1.block.0.weight", "res_blocks.1.block.0.bias", "res_blocks.1.block.1.weight", "res_blocks.1.block.1.bias",
    "res_blocks.1.block.3.weight", "res_blocks.1.block.3.bias", "res_blocks.1.block.4.weight", "res_blocks.1.block.4.bias",
    "grid_fc.0.weight", "grid_fc.0.bias",
    "features_preprocess_fc.0.weight", "features_preprocess_fc.0.bias",
    "combined_fc.0.weight", "combined_fc.0.bias",
]


def grouped_extractor_forward(p: StackedParameters, idx: torch.Tensor, grid: torch.Tensor, features: torch.Tensor,
                              prefix: str = EXTRACTOR_PREFIX) -> torch.Tensor:
    def w(name):
        return p[prefix + name]

    x = grouped_conv3x3(grid, w("stem.0.weight"), w("stem.0.bias"), idx)
    x = F.relu(grouped_group_norm(x, w("stem.1.weight"), w("stem.1.bias"), idx))
    for block in ("res_blocks.0.block.", "res_blocks.1.block."):
        y = grouped_conv3x3(x, w(block + "0.weight"), w(block + "0.bias"), idx)
        y = F.relu(grouped_group_norm(y, w(block + "1.weight"), w(block + "1.bias"), idx))
        y = grouped_conv3x3(y, w(block + "3.weight"), w(block + "3.bias"), idx)
        y = grouped_group_norm(y, w(block + "4.weight"), w(block + "4.bias"), idx)
        x = F.relu(x + y)
    x = F.adaptive_avg_pool2d(x, (3, 3)).flatten(1)
    g = F.relu(grouped_linear(x, w("grid_fc.0.weight"), w("grid_fc.0.bias"), idx))
    f = F.relu(grouped_linear(features, w("features_preprocess_fc.0.weight"), w("features_preprocess_fc.0.bias"), idx))
    return F.relu(grouped_linear(torch.cat([g, f], dim=1), w("combined_fc.0.weight"), w("combined_fc.0.bias"), idx))


def observation_tensors(obs: dict) -> tuple:
    grid = obs["grid_tensor"]
    grid_t = grid if torch.is_tensor(grid) else torch.from_numpy(np.ascontiguousarray(grid))
    feats = obs["features"]
    feats_t = feats if torch.is_tensor(feats) else torch.from_numpy(np.ascontiguousarray(feats))
    return grid_t.to(torch.float32), feats_t.to(torch.float32)


def model_indices(paths: List[str], index: Dict[str, int]) -> torch.Tensor:
    return torch.from_numpy(np.fromiter((index[p] for p in paths), dtype=np.int64, count=len(paths)))
