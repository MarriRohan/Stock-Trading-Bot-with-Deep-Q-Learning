##**RL-Based Stop-Loss and Take-Profit Trading Bot**##

A reinforcement learning-driven trading system that learns optimal Stop-Loss, Take-Profit, and Trailing Stop strategies across different market conditions, while enforcing strict capital and risk management.

**1. Project Overview**

This project implements a Deep Q-Learning (DQN) based trading bot that:

Learns dynamic Stop-Loss, Take-Profit, and Trailing Stop values

Adapts its strategy across different market regimes (bullish, bearish, sideways)

Enforces a maximum capital limit to prevent excessive loss or misuse of funds

Applies risk management measures such as daily loss limits and drawdown protection

Supports alert mechanisms (email, Telegram, Slack) and simulated fund withdrawal/top-up actions

Works in backtesting and paper trading mode only (safe, no real capital required)

**2. How It Works**
Reinforcement Learning Framework
Component	Description
State	OHLCV price data, technical indicators (ATR, RSI, returns), wallet equity, unrealized PnL, drawdown, regime ID
Action	Selects a combination of Stop-Loss, Take-Profit, and Trailing Stop from a discrete grid (e.g., SL = 0.5/0.75/1.0 × ATR, TP = 1.0/1.5/2.0 × ATR, Trailing = 0.5/0.75/1.0 × ATR)
Reward	Profit or loss per step, minus transaction costs and penalties for large drawdown or risky trades
Episode End	End of dataset or when max daily loss or drawdown threshold is breached (circuit breaker)

**3. Risk and Capital Management**

Maximum capital limit: Bot cannot exceed this value in total market exposure

Daily loss limit: Automatically halts trading if exceeded

Position size = (Risk per trade) / (Stop-Loss distance)

No leverage, no martingale, no averaging into losing positions

Optional simulated withdrawal or capital reduction if losses exceed a critical level
