#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Chengyang Gu"
__email__ = "chengyang.gu@nyu.edu"
__version__ = "1"

import torch

class FullyConnectedNeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2, 11),
            torch.nn.BatchNorm1d(11),
            torch.nn.ELU(),
            torch.nn.Linear(11, 11),
            torch.nn.BatchNorm1d(11),
            torch.nn.ELU(),
            torch.nn.Linear(11, 11),
            torch.nn.BatchNorm1d(11),
            torch.nn.ELU(),
            torch.nn.Linear(11, 1)
        )

    def forward(self, x):
        return self.fc(x)
