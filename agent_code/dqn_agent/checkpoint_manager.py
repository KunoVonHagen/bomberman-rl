from __future__ import annotations

import json
import pathlib
from typing import Optional

from agent_code.ppo_agent.checkpoint_manager import CheckpointManager as _BaseCheckpointManager
from .config import TrainingConfig


class CheckpointManager(_BaseCheckpointManager):
    """CheckpointManager bound to the DQN TrainingConfig instead of the PPO one."""

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