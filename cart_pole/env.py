from copy import deepcopy
from typing import Tuple

import gym
import numpy as np
from gym.wrappers import TimeLimit


class CartPole(gym.Env):
    """
    Wrapper for gym cart_pole environment where the reward
    is accumulated to the end
    """

    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.running_reward = 0
        self.score = 0

    def reset(self):
        self.running_reward = 0
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.running_reward += reward
        self.score = self.running_reward if done else 0
        return (
            obs,
            self.score,
            done,
            info,
        )

    def set_state(self, state: Tuple[TimeLimit, float]):
        self.running_reward = state[1]
        self.env = deepcopy(state[0])
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env), self.running_reward, self.score
