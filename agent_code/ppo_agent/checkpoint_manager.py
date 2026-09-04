from __future__ import annotations

import json
import copy
import pathlib
from datetime import datetime
from typing import Optional, List
import numpy as np

from .config import TrainingConfig


def _json_safe(obj):
    """json.dumps(default=...) fallback for common non-serializable types
    that sneak into metadata dicts (numpy scalars/arrays, pathlib.Path)."""

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class CheckpointManager:
    def __init__(self, run_dir: pathlib.Path, config: TrainingConfig):
        self.run_dir = pathlib.Path(run_dir)
        self.config = config

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.replays_dir = self.run_dir / "replays"
        self.logs_dir = self.run_dir / "logs"

        for d in (self.checkpoints_dir, self.tensorboard_dir, self.replays_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def new(cls, config: TrainingConfig, architecture_info: Optional[dict] = None) -> "CheckpointManager":
        run_name = config.run_name or f"run_{datetime.now():%Y%m%d-%H%M%S}"
        config = copy.deepcopy(config)
        config.run_name = run_name

        run_dir = pathlib.Path(config.runs_dir) / run_name
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory '{run_dir}' already exists. Pick a different "
                f"run_name in your config, or use CheckpointManager.resume()."
            )
        run_dir.mkdir(parents=True)

        manager = cls(run_dir, config)
        manager._write_manifest(architecture_info or {})
        return manager

    @classmethod
    def resume(
        cls,
        run_name_or_path: str,
        runs_dir: str = "runs",
        config_overrides: Optional[dict] = None,
    ) -> "CheckpointManager":
        run_dir = pathlib.Path(run_name_or_path)
        if not run_dir.exists():
            run_dir = pathlib.Path(runs_dir) / run_name_or_path
        if not run_dir.exists():
            raise FileNotFoundError(f"No run found at '{run_name_or_path}'")

        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"'{run_dir}' has no run_manifest.json — not a valid run folder")

        manifest = json.loads(manifest_path.read_text())
        config = TrainingConfig.from_dict(manifest["config"])

        if config_overrides:
            for k, v in config_overrides.items():
                setattr(config, k, v)

        return cls(run_dir, config)

    def _write_manifest(self, architecture_info: dict) -> None:
        manifest = {
            "run_name": self.config.run_name,
            "created_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "architecture": architecture_info,
        }
        (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def update_manifest_config(self, architecture_info: Optional[dict] = None) -> None:
        """
        Update the run_manifest.json with the current config and architecture info.
        If architecture_info is None, it will be read from the existing manifest.
        """
        info = architecture_info
        if info is None:
            existing = self.read_manifest() if self.manifest_path.exists() else {}
            info = existing.get("architecture", {})
        self._write_manifest(info)

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.run_dir / "run_manifest.json"

    def read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text())

    def _checkpoint_dir(self, timesteps: int) -> pathlib.Path:
        return self.checkpoints_dir / f"checkpoint_{timesteps:010d}"

    def save_checkpoint(
        self,
        model,
        timesteps: int,
        extra_metadata: Optional[dict] = None,
    ) -> pathlib.Path:
        ckpt_dir = self._checkpoint_dir(timesteps)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        model.save(str(ckpt_dir / "model.zip"))

        metadata = {"timesteps": timesteps, "saved_at": datetime.now().isoformat()}
        if extra_metadata:
            metadata.update(extra_metadata)
        (ckpt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=_json_safe))

        (self.checkpoints_dir / "latest.txt").write_text(ckpt_dir.name)

        return ckpt_dir

    def list_checkpoints(self) -> List[pathlib.Path]:
        ckpts = [p for p in self.checkpoints_dir.glob("checkpoint_*") if p.is_dir()]
        return sorted(ckpts, key=lambda p: int(p.name.split("_")[-1]))

    def latest_checkpoint(self) -> Optional[pathlib.Path]:
        pointer = self.checkpoints_dir / "latest.txt"
        if pointer.exists():
            path = self.checkpoints_dir / pointer.read_text().strip()
            if path.exists():
                return path
        ckpts = self.list_checkpoints()
        return ckpts[-1] if ckpts else None

    def get_checkpoint(self, name_or_timesteps) -> pathlib.Path:
        path = (
            self._checkpoint_dir(name_or_timesteps)
            if isinstance(name_or_timesteps, int)
            else self.checkpoints_dir / str(name_or_timesteps)
        )
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    def load_model(self, model_cls, checkpoint: Optional[pathlib.Path] = None, env=None, **load_kwargs):
        """
        Load a model from a checkpoint dir (defaults to the latest one).
        Returns model.
        """
        ckpt_dir = checkpoint or self.latest_checkpoint()
        if ckpt_dir is None:
            return None, None
        model = model_cls.load(str(ckpt_dir / "model.zip"), env=env, **load_kwargs)
        return model

    @staticmethod
    def resolved_timesteps(checkpoint_dir: pathlib.Path) -> int:
        return int(checkpoint_dir.name.split("_")[-1])