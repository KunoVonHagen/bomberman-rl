from __future__ import annotations

import copy
import json
import pathlib
from datetime import datetime
from typing import Optional, List

import numpy as np

from .config import TrainingConfig


def to_json_compatible(obj):
    """Convert common runtime values to JSON-safe types."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pathlib.Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class CheckpointManager:
    """Create and restore training checkpoints for a run."""

    def __init__(self, run_dir: pathlib.Path, config: TrainingConfig):
        self.run_dir = pathlib.Path(run_dir)
        self.config = config

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.tensorboard_dir = self.run_dir / "tensorboard"
        self.replays_dir = self.run_dir / "replays"
        self.logs_dir = self.run_dir / "logs"

        for directory in (self.checkpoints_dir, self.tensorboard_dir, self.replays_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def new(cls, config: TrainingConfig, architecture_info: Optional[dict] = None) -> "CheckpointManager":
        """Create a new run directory and manifest."""
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
        """Resume a previous run from disk."""
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
            for key, value in config_overrides.items():
                setattr(config, key, value)

        return cls(run_dir, config)

    def _write_manifest(self, architecture_info: dict) -> None:
        """Persist the run manifest to disk."""
        manifest = {
            "run_name": self.config.run_name,
            "created_at": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "architecture": architecture_info,
        }
        (self.run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    def update_manifest_config(self, architecture_info: Optional[dict] = None) -> None:
        """Refresh the manifest with the current configuration."""
        info = architecture_info
        if info is None:
            existing = self.read_manifest() if self.manifest_path.exists() else {}
            info = existing.get("architecture", {})
        self._write_manifest(info)

    @property
    def manifest_path(self) -> pathlib.Path:
        return self.run_dir / "run_manifest.json"

    def read_manifest(self) -> dict:
        """Load the run manifest from disk."""
        return json.loads(self.manifest_path.read_text())

    def _checkpoint_dir(self, timesteps: int) -> pathlib.Path:
        return self.checkpoints_dir / f"checkpoint_{timesteps:010d}"

    def save_checkpoint(
        self,
        model,
        timesteps: int,
        extra_metadata: Optional[dict] = None,
    ) -> pathlib.Path:
        """Save a model checkpoint and its metadata."""
        checkpoint_dir = self._checkpoint_dir(timesteps)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for leaky_attr in ("train", "collect_rollouts", "predict"):
            model.__dict__.pop(leaky_attr, None)

        model.save(str(checkpoint_dir / "model.zip"))

        metadata = {"timesteps": timesteps, "saved_at": datetime.now().isoformat()}
        if extra_metadata:
            metadata.update(extra_metadata)
        (checkpoint_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, default=to_json_compatible)
        )

        (self.checkpoints_dir / "latest.txt").write_text(checkpoint_dir.name)
        return checkpoint_dir

    def list_checkpoints(self) -> List[pathlib.Path]:
        """Return all saved checkpoints in chronological order."""
        checkpoints = [path for path in self.checkpoints_dir.glob("checkpoint_*") if path.is_dir()]
        return sorted(checkpoints, key=lambda path: int(path.name.split("_")[-1]))

    def latest_checkpoint(self) -> Optional[pathlib.Path]:
        """Return the newest checkpoint if one exists."""
        pointer = self.checkpoints_dir / "latest.txt"
        if pointer.exists():
            path = self.checkpoints_dir / pointer.read_text().strip()
            if path.exists():
                return path
        checkpoints = self.list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def get_checkpoint(self, name_or_timesteps) -> pathlib.Path:
        """Resolve a checkpoint path from a name or timestep."""
        path = (
            self._checkpoint_dir(name_or_timesteps)
            if isinstance(name_or_timesteps, int)
            else self.checkpoints_dir / str(name_or_timesteps)
        )
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    def load_model(self, model_cls, checkpoint: Optional[pathlib.Path] = None, env=None, **load_kwargs):
        """Load a model from a checkpoint directory."""
        checkpoint_dir = checkpoint or self.latest_checkpoint()
        if checkpoint_dir is None:
            return None, None
        model = model_cls.load(str(checkpoint_dir / "model.zip"), env=env, **load_kwargs)
        return model

    @staticmethod
    def resolved_timesteps(checkpoint_dir: pathlib.Path) -> int:
        """Read the timestep count from a checkpoint directory name."""
        return int(checkpoint_dir.name.split("_")[-1])

