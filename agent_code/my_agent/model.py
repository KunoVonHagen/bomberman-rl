import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class BombermanFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 32):
        super().__init__(observation_space, features_dim)
        n_layers = observation_space.shape[0]

        #self.mix = nn.Sequential(nn.Conv2d(n_layers, 64, kernel_size=1), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(n_layers, 32, kernel_size=3, padding=1), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            flat_dim = self._forward_conv(sample).shape[1]

        print(flat_dim)
        print(flat_dim*features_dim)

        self.fc = nn.Sequential(nn.Linear(flat_dim, features_dim), nn.ReLU())
        self._init_weights()

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x

    def forward(self, observations):
        return self.fc(self._forward_conv(observations))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.zeros_(m.bias)