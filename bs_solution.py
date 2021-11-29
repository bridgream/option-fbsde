#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

__author__ = "Chengyang Gu"
__email__ = "chengyang.gu@nyu.edu"
__version__ = "1"

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

risk_free_rate = 0.06
volatility = 0.2
long_call_strike = 120
short_call_strike = 150
time = 0.5  # in years
n_time_step = 50

def black_scholes_call_price(s0, k, t, r, sigma):
    d1 = 1 / (sigma * np.sqrt(t)) * (np.log(s0 / k) + (r + sigma**2 / 2) * t)
    d2 = d1 - sigma * np.sqrt(t)
    return scipy.stats.norm.cdf(d1) * s0 - scipy.stats.norm.cdf(d2) * k * np.exp(-r * t)


def price_portfolio(_p0):
    """
    A portfolio of one long call @ 120 and two short call @ 150
    :param _p0: current price
    :return: portfolio value
    """
    return (
            black_scholes_call_price(_p0, long_call_strike, time, risk_free_rate, volatility)
            - 2 * black_scholes_call_price(_p0, short_call_strike, time, risk_free_rate, volatility)
    )

start_price_grid = np.linspace(70, 170, 1000)
plt.plot(start_price_grid, price_portfolio(start_price_grid), label="Analytical Solution (BS)")
plt.legend()
plt.xlabel("Start Underlying Price")
plt.ylabel("Portfolio Value")
plt.grid()
plt.savefig("fig/solution")
