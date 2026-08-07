import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, H, W)
        weights = self.fc(x.mean(dim=(2, 3)))  # (B, C)
        return x * weights.unsqueeze(-1).unsqueeze(-1)


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, H, W)
        b, c, h, w = x.shape
        tokens = self.norm(x).flatten(2).transpose(1, 2)  # (B, H*W, C)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        attn_out = attn_out.transpose(1, 2).view(b, c, h, w)
        return x + attn_out  # residual


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

        self.register_buffer(
            "coord_x",
            torch.linspace(-1, 1, w).view(1, 1, 1, w).expand(1, 1, h, w).clone(),
        )
        self.register_buffer(
            "coord_y",
            torch.linspace(-1, 1, h).view(1, 1, h, 1).expand(1, 1, h, w).clone(),
        )

        self.channel_attn = ChannelAttention(n_input_channels + 2)

        stem_channels = 64
        self.stem = nn.Sequential(
            nn.Conv2d(n_input_channels + 2, stem_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, stem_channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(stem_channels),
            ResidualBlock(stem_channels),
            ResidualBlock(stem_channels),
        )

        self.spatial_attn = SpatialSelfAttention(stem_channels, num_heads=4)

        self.pool = nn.AdaptiveAvgPool2d((3, 3))
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample = torch.zeros(1, n_input_channels, h, w, dtype=torch.float32)
            n_cnn_flatten = self._conv_forward(sample).shape[1]

        self.grid_fc = nn.Sequential(
            nn.Linear(n_cnn_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.features_preprocess_fc = nn.Sequential(
            nn.Linear(feature_space.shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.combined_fc = nn.Sequential(
            nn.Linear(128 + 64, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
        )

    def _conv_forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        coord_x = self.coord_x.expand(batch, -1, -1, -1)
        coord_y = self.coord_y.expand(batch, -1, -1, -1)
        x = torch.cat([x, coord_x, coord_y], dim=1)
        x = self.channel_attn(x)
        x = self.stem(x)
        x = self.res_blocks(x)
        x = self.spatial_attn(x)
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