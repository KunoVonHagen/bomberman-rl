import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
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


class BombermanFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 256):

        super().__init__(observation_space, features_dim)

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
        )

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU(),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.pool(x)
        return self.flatten(x)

    def forward(self, observations: dict) -> torch.Tensor:
        grid_tensor = observations["grid_tensor"].float()  # (B, C, H, W)
        features = observations["features"].float()        # (B, F)

        cnn_out = self._conv_forward(grid_tensor)
        cnn_fc_out = self.grid_fc(cnn_out)

        features_fc_out = self.features_preprocess_fc(features)

        combined = torch.cat([cnn_fc_out, features_fc_out], dim=1)
        combined_out = self.combined_fc(combined)

        return combined_out