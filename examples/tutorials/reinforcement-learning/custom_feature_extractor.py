from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import gym
import torch as th

class CustomExtracor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Space, features_dim: int):
        self.flattened_dim = get_flattened_obs_dim(observation_space)
        super().__init__(observation_space, features_dim=features_dim)
        self.flatten = th.nn.Flatten()
        self.fc1 = th.nn.Linear(self.flattened_dim, self.flattened_dim)
        self.activation_fn = th.nn.SELU()
        self.fc2 = th.nn.Linear(self.flattened_dim, self.features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        flattened_obs = self.flatten(observations)
        features = self.fc2(self.activation_fn(self.fc1(flattened_obs)))
        return features