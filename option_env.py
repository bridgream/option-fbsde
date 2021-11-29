#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Chengyang Gu"
__email__ = "chengyang.gu@nyu.edu"
__version__ = "1"

import gym
import numpy as np


class OptionEnv(gym.Env):
    def __init__(self):
        self.s0 = 120
        self.st = self.s0
        self.time = 0.5  # in years
        self.episode_length = 50
        self.risk_free_rate = 0.06
        self.volatility = 0.2

        self.dt = self.time / self.episode_length
        self.sqrt_dt = np.sqrt(self.dt)

        self._i_step = 0

        self.action_space = gym.spaces.Box(np.asarray([0.]), np.asarray([1.]))
        self.observation_space = gym.spaces.Box(np.asarray([0.]), np.asarray([np.inf]))

    def step(self, action: np.ndarray):
        portfolio_value = action.item()

        d_st = self.st * (self.risk_free_rate * self.dt + self.volatility * np.random.rand() * self.sqrt_dt)
        reward =
        self.st += d_st

        self._i_step += 1

        done = (self._i_step == self.episode_length)

        return self.st, reward, done, {}

    def reset(self):
        self.st = self.s0
        self._i_step = 0

        return self.s0

    def render(self, mode="human"):
        pass