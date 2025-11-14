import numpy as np


class TradingEnv:
"""Minimal gym-like trading environment with capital/risk enforcement.
Observation is a fixed-size vector (for demo). Actions pick an index mapping to SL/TP/trailing combos.
"""
def __init__(self, data, config):
self.data = data
self.cfg = config
self.max_capital = config.get('max_capital', 100000)
self.daily_loss_limit = config.get('daily_loss_limit', 0.02)
self.drawdown_limit = config.get('drawdown_limit', 0.15)
self.reset()


def reset(self):
self.wallet = self.max_capital
self.equity_peak = self.wallet
self.position = None
self.position_price = 0.0
self.current_step = 0
self.daily_start_equity = self.wallet
self.done = False
return self._get_obs()


def step(self, action):
# action is an int index, map to sl/tp/trailing externally
reward = 0.0
info = {}


# advance price (use close price)
price = self.data[self.current_step]


# very simple position logic: either flat or long one unit
if self.position is None:
# open position
self.position = 1
self.position_price = price
else:
# evaluate PnL and close every N steps for demo
pnl = price - self.position_price
self.wallet += pnl
reward = pnl
self.position = None
self.position_price = 0.0


# enforce daily loss limit
daily_loss = (self.daily_start_equity - self.wallet) / max(1, self.daily_start_equity)
if daily_loss >= self.daily_loss_limit:
info['circuit'] = 'daily_loss_limit'
self.done = True


# enforce drawdown
if self.wallet > self.equity_peak:
self.equity_peak = self.wallet
drawdown = (self.equity_peak - self.wallet) / max(1, self.equity_peak)
if drawdown >= self.drawdown_limit:
info['circuit'] = 'drawdown_limit'
self.done = True


self.current_step += 1
if self.current_step >= len(self.data):
self.done = True


return self._get_obs(), reward, self.done, info


def _get_obs(self):
# return a fixed-size vector: [price, wallet, equity_peak, step_norm]
p = float(self.data[self.current_step]) if self.current_step < len(self.data) else float(self.data[-1])
return np.array([p, self.wallet, self.equity_peak, self.current_step/len(self.data)], dtype=np.float32)
