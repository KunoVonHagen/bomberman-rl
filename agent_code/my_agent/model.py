import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class MixBlock(nn.Module):
    def __init__(self, n_layers, n_channels):
        super(MixBlock, self).__init__()
        self.conv = nn.Conv2d(n_layers, n_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()


    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out) + x)
        return out


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class FeatureMLP(nn.Module):
    """Small MLP branch for the low-dimensional global scalar features
    (`observations["features"]`). These are cheap, always-populated,
    board-independent scalars (agent position, bomb danger, distances,
    remaining coins/crates/opponents, etc.) -- a couple of conv layers
    would be massive overkill and would only dilute the signal, so they
    get their own lightweight path instead of being stacked as constant
    planes into the CNN input."""

    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class BombermanFeatureExtractor(BaseFeaturesExtractor):
    """Fuses the spatial `grid_tensor` (via a small residual CNN) with the
    global scalar `features` vector (via a small MLP) into a single
    `features_dim`-sized embedding for the SB3 policy/value heads.

    Expects a `gym.spaces.Dict` observation space with two keys:
      - "grid_tensor": Box(n_layers, H, W) -- the spatial layers.
      - "features":    Box(n_scalar_features,) -- the always-on global
        scalar features (see `BombermanGymEnv.feature_names()`).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 64,
        feature_embed_dim: int = 32,
    ):
        super().__init__(observation_space, features_dim)

        grid_space = observation_space["grid_tensor"]
        feat_space = observation_space["features"]

        n_layers = grid_space.shape[0]
        n_scalar_features = feat_space.shape[0]

        self.mix = MixBlock(n_layers, 64)
        self.residual1 = ResidualBlock(64)
        self.residual2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        with torch.no_grad():
            sample = torch.zeros(1, *grid_space.shape)
            conv_flat_dim = self._forward_conv(sample).shape[1]

        self.feature_mlp = FeatureMLP(n_scalar_features, feature_embed_dim)

        self.fc = nn.Sequential(
            nn.Linear(conv_flat_dim + feature_embed_dim, features_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _forward_conv(self, x):
        x = self.mix(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def forward(self, observations):
        conv_out = self._forward_conv(observations["grid_tensor"])
        feat_out = self.feature_mlp(observations["features"])
        combined = torch.cat([conv_out, feat_out], dim=1)
        return self.fc(combined)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)