import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BombermanFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,
                 features_dim: int = 128):

        super().__init__(observation_space, features_dim)

        grid_space = observation_space["grid_tensor"]
        feature_space = observation_space["features"]

        n_input_channels = grid_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4,4)),

            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.zeros(1, *grid_space.shape, dtype=torch.float32)
            n_cnn_flatten = self.cnn(sample).shape[1]

        self.grid_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten, 128),
            nn.ReLU(),
        )

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU()
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU()
        )

    def _conv_forward(self, x):
        return self.cnn(x)

    def forward(self, observations: dict) -> torch.Tensor:
        grid_tensor = observations["grid_tensor"].float()  # (B, C, H, W)
        features = observations["features"].float() # (B, F)

        cnn_out = self.cnn(grid_tensor)
        cnn_fc_out = self.grid_fc(cnn_out)

        features_fc_out = self.features_preprocess_fc(features)

        combined = torch.cat([cnn_fc_out, features_fc_out], dim=1)
        combined_out = self.combined_fc(combined)

        return combined_out