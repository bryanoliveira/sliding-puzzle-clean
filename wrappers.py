import gym
import numpy as np


class ChannelFirstImageWrapper(gym.ObservationWrapper):
    def __init__(self, env, **kwargs):
        super().__init__(env)
        assert env.observation_space.shape == 3
        assert env.observation_space.dtype == np.uint8
        channel_first_shape = sorted(env.observation_space.shape)
        self.should_transpose = channel_first_shape != env.observation_space.shape
        
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=channel_first_shape,
            dtype=np.uint8,
        )

    def observation(self, obs):
        if self.should_transpose:
            return obs.transpose(2, 0, 1)
        return obs