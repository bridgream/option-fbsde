#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Chengyang Gu"
__email__ = "chengyang.gu@nyu.edu"
__version__ = "1"

import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.tensorboard
from tqdm import tqdm

from fcnn import FullyConnectedNeuralNetwork

def reward_function(price):
    return torch.clip(price - 120, min=0) - 2 * torch.clip(price - 150, min=0)

risk_free_rate = 0.06
volatility = 0.2
long_call_strike = 120
short_call_strike = 150
time = 0.5  # in years
n_time_step = 50
dt = time / n_time_step
sqrt_dt = np.sqrt(dt)

n_iter = 20000
batch_size = 512
loss_func = torch.nn.MSELoss()
tb_writer = torch.utils.tensorboard.SummaryWriter()

model = FullyConnectedNeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
y_0 = torch.nn.Parameter(torch.ones(1, requires_grad=True))
optimizer.add_param_group({"params": y_0})

z_0 = torch.ones(1, requires_grad=True)

for i_iter in tqdm(range(n_iter)):
    x = torch.zeros(batch_size, 1) + 120
    y = torch.zeros_like(x) + y_0

    for t in np.linspace(0, 1, n_time_step + 1):
        z = model(torch.concat([x, torch.zeros_like(x) + t], dim=1))
        dw = torch.randn_like(x)
        dx = x * (risk_free_rate * dt + torch.randn_like(x) * volatility * sqrt_dt)
        dy = risk_free_rate * y * dt + volatility * x * z * dw
        x += dx
        y += dy

    loss = loss_func(y, reward_function(x))
    tb_writer.add_scalar("Loss", loss.item(), i_iter)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    tb_writer.add_scalar("Y_0", y_0.item(), i_iter)

with open("model.pkl", "wb") as writer:
    pickle.dump(model, writer)
