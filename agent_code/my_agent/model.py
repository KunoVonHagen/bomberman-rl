import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FeatureCompression(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureCompression, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.conv(self.relu(x))

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))

"""
class SpatialAttention(nn.Module):
    def __init__(
            self,
            channels,
            num_tokens,
            nhead=4,
            dim_feedforward=128,
            num_layers=2
    ):
        super(SpatialAttention, self).__init__()

        self.position = nn.Parameter(
            torch.randn(1, num_tokens, channels)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


    def forward(self, x):
        # x: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.size()
        num_tokens = height * width

        # Reshape to (batch_size, num_tokens, channels)
        x = x.view(batch_size, channels, num_tokens).permute(0, 2, 1)

        # Add positional encoding
        x = x + self.position

        # Pass through transformer encoder
        out = self.transformer(x)

        # Global board embedding
        out = out.mean(dim=1)

        return out
"""


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        return self.relu(self.conv2(out) + identity)


class SpatialEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SpatialEncoder, self).__init__()

        self.feature_compression = FeatureCompression(in_channels, out_channels // 4)
        self.conv1 = nn.Conv2d(out_channels // 4, out_channels // 2, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(out_channels // 2)
        self.downsample1 = DownsamplingBlock(out_channels // 2, out_channels)
        self.res2 = ResidualBlock(out_channels)

    def forward(self, x):
        out = self.feature_compression(x)
        out = self.conv1(out)
        out = self.res1(out)
        out = self.downsample1(out)
        out = self.res2(out)

        return out


class BombermanFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, out_features=128):
        super(BombermanFeatureExtractor, self).__init__(observation_space, out_features)


        self.board_channels = observation_space["grid_tensor"].shape[0]
        self.board_height = observation_space["grid_tensor"].shape[1]
        self.board_width = observation_space["grid_tensor"].shape[2]
        self.numeric_features = observation_space["features"].shape[0]
        self.out_features = out_features

        self.spatial_encoder = SpatialEncoder(self.board_channels, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)

        """
        with torch.no_grad():
            test_tensor = torch.zeros(1, self.board_channels, self.board_height, self.board_width)
            _, _, H, W = self.spatial_encoder(test_tensor).shape
            num_tokens = H * W


        self.spatial_attention = SpatialAttention(64, num_tokens)
        """

        self.fc = nn.Sequential(
            nn.Linear(64 + self.numeric_features, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, self.out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, observations):
        grid_tensor = observations["grid_tensor"]
        features = observations["features"]

        spatial_features = self.spatial_encoder(grid_tensor)
        spatial_features = torch.flatten(self.pool(spatial_features), 1)

        # attention_features = self.spatial_attention(spatial_features)

        combined_features = torch.cat([spatial_features, features], dim=1)
        out = self.fc(combined_features)

        return out
