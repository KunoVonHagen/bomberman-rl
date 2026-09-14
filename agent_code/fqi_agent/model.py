from __future__ import annotations

import pathlib

import numpy as np
from numba import njit, prange

MODEL_FILE = "model.npz"


def masked_greedy(q_values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    q = np.where(masks, q_values, -np.inf)
    return np.argmax(q, axis=-1)


def masked_max(q_values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    q = np.where(masks, q_values, -np.inf)
    best = q.max(axis=-1)
    return np.where(np.isfinite(best), best, 0.0)


class LinearQ:
    kind = "linear"

    def __init__(self, n_features: int, n_actions: int, expansion: str = "linear", ridge: float = 1e-2):
        if expansion not in ("linear", "quadratic"):
            raise ValueError(f"unknown expansion {expansion!r}")
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.expansion = expansion
        self.ridge = float(ridge)
        self.pairs_i, self.pairs_j = np.triu_indices(self.n_features)
        self.weights = np.zeros((self.n_actions, self.n_parameters), dtype=np.float64)

    @property
    def n_parameters(self) -> int:
        n = 1 + self.n_features
        if self.expansion == "quadratic":
            n += self.n_features * (self.n_features + 1) // 2
        return n

    def expand(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1, self.n_features)
        columns = [np.ones((x.shape[0], 1)), x]
        if self.expansion == "quadratic":
            columns.append(x[:, self.pairs_i] * x[:, self.pairs_j])
        return np.concatenate(columns, axis=1)

    def fit(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray, seed: int = 0) -> dict:
        phi = self.expand(x)
        actions = np.asarray(actions).reshape(-1)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1)
        counts = {}
        for a in range(self.n_actions):
            rows = actions == a
            counts[a] = int(rows.sum())
            if counts[a] == 0:
                continue
            p = phi[rows]
            eigenvalues, eigenvectors = np.linalg.eigh(p.T @ p)
            projected = eigenvectors.T @ (p.T @ targets[rows])
            self.weights[a] = eigenvectors @ (projected / (eigenvalues + self.ridge))
        return counts

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.expand(x) @ self.weights.T).astype(np.float32)

    def state(self) -> dict:
        return dict(kind=self.kind, n_features=self.n_features, n_actions=self.n_actions,
                    expansion=self.expansion, ridge=self.ridge, weights=self.weights)

    @classmethod
    def from_state(cls, data) -> "LinearQ":
        model = cls(int(data["n_features"]), int(data["n_actions"]), str(data["expansion"]), float(data["ridge"]))
        model.weights = np.asarray(data["weights"], dtype=np.float64)
        return model


@njit(cache=True, parallel=True)
def _forest_predict(x, left, right, feature, threshold, value, tree_start, out):
    n = x.shape[0]
    n_trees = tree_start.shape[0]
    for i in prange(n):
        total = 0.0
        for t in range(n_trees):
            node = tree_start[t]
            while left[node] >= 0:
                if x[i, feature[node]] <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
            total += value[node]
        out[i] = total / n_trees


_TREE_KEYS = ("left", "right", "feature", "threshold", "value", "tree_start")


class ForestQ:
    kind = "forest"

    def __init__(self, n_features: int, n_actions: int, n_estimators: int = 50, max_depth: int = 12,
                 min_samples_leaf: int = 20, max_features: int = 10, n_jobs: int = -1):
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = int(max_features)
        self.n_jobs = int(n_jobs)
        self.trees = [None] * self.n_actions

    def fit(self, x: np.ndarray, actions: np.ndarray, targets: np.ndarray, seed: int = 0) -> dict:
        from sklearn.ensemble import RandomForestRegressor

        x = np.asarray(x, dtype=np.float32).reshape(-1, self.n_features)
        actions = np.asarray(actions).reshape(-1)
        targets = np.asarray(targets, dtype=np.float32).reshape(-1)
        counts = {}
        for a in range(self.n_actions):
            rows = actions == a
            counts[a] = int(rows.sum())
            if counts[a] == 0:
                continue
            forest = RandomForestRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf, max_features=min(self.max_features, self.n_features),
                n_jobs=self.n_jobs, random_state=seed + a,
            )
            forest.fit(x[rows], targets[rows])
            self.trees[a] = _export_forest(forest)
        return counts

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32).reshape(-1, self.n_features))
        out = np.zeros((x.shape[0], self.n_actions), dtype=np.float32)
        for a, trees in enumerate(self.trees):
            if trees is None:
                continue
            column = np.zeros(x.shape[0], dtype=np.float64)
            _forest_predict(x, trees["left"], trees["right"], trees["feature"], trees["threshold"],
                            trees["value"], trees["tree_start"], column)
            out[:, a] = column
        return out

    def state(self) -> dict:
        data = dict(kind=self.kind, n_features=self.n_features, n_actions=self.n_actions,
                    n_estimators=self.n_estimators, max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf, max_features=self.max_features,
                    fitted=np.array([t is not None for t in self.trees]))
        for a, trees in enumerate(self.trees):
            if trees is None:
                continue
            for key in _TREE_KEYS:
                data[f"{key}_{a}"] = trees[key]
        return data

    @classmethod
    def from_state(cls, data) -> "ForestQ":
        model = cls(int(data["n_features"]), int(data["n_actions"]), int(data["n_estimators"]),
                    int(data["max_depth"]), int(data["min_samples_leaf"]), int(data["max_features"]))
        for a, fitted in enumerate(np.asarray(data["fitted"])):
            if fitted:
                model.trees[a] = {key: np.ascontiguousarray(data[f"{key}_{a}"]) for key in _TREE_KEYS}
        return model


def _export_forest(forest) -> dict:
    left, right, feature, threshold, value, starts = [], [], [], [], [], []
    offset = 0
    for estimator in forest.estimators_:
        tree = estimator.tree_
        starts.append(offset)
        left.append(np.where(tree.children_left >= 0, tree.children_left + offset, -1))
        right.append(np.where(tree.children_right >= 0, tree.children_right + offset, -1))
        feature.append(np.maximum(tree.feature, 0))
        threshold.append(tree.threshold)
        value.append(tree.value[:, 0, 0])
        offset += tree.node_count
    return dict(
        left=np.concatenate(left).astype(np.int64),
        right=np.concatenate(right).astype(np.int64),
        feature=np.concatenate(feature).astype(np.int64),
        threshold=np.concatenate(threshold).astype(np.float64),
        value=np.concatenate(value).astype(np.float64),
        tree_start=np.asarray(starts, dtype=np.int64),
    )


def build_model(kind: str, n_features: int, n_actions: int, **kwargs):
    if kind in ("linear", "quadratic"):
        return LinearQ(n_features, n_actions, expansion=kind, ridge=kwargs["ridge"])
    if kind == "forest":
        return ForestQ(n_features, n_actions, n_estimators=kwargs["n_estimators"], max_depth=kwargs["max_depth"],
                       min_samples_leaf=kwargs["min_samples_leaf"], max_features=kwargs["max_features"])
    raise ValueError(f"unknown model kind {kind!r}")


def save_model(model, path) -> None:
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **model.state())
    tmp.replace(path)


def load_model(path):
    with np.load(path) as data:
        kind = str(data["kind"])
        if kind == LinearQ.kind:
            return LinearQ.from_state(data)
        if kind == ForestQ.kind:
            return ForestQ.from_state(data)
    raise ValueError(f"unknown model kind {kind!r} in {path}")
