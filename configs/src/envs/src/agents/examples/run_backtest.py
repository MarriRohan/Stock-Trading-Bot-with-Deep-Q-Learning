# examples/run_backtest.py
import os
import yaml
import numpy as np
from src.envs.trading_env import TradingEnv
from src.agents.dqn_agent import DQNAgent

def load_config(path='configs/example.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_toy_prices(length=1000):
    # simple sine-wave + noise as toy price series
    x = np.linspace(0, 20, length)
    prices = (np.sin(x) * 10.0 + 100.0) + np.random.normal(scale=0.5, size=length)
    return prices.tolist()

def main():
    cfg = load_config()
    prices = generate_toy_prices(1000)

    env = TradingEnv(prices, cfg)
    state_dim = 4
    action_dim = 9  # placeholder for grid size (sl * tp * trailing)
    agent = DQNAgent(state_dim, action_dim, cfg)

    episodes = cfg.get('episodes', 10)
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        while not done:
            a = agent.act(s)
            s2, r, done, info = env.step(a)
            agent.remember(s, a, r, s2, done)
            loss = agent.learn()
            s = s2
            total_reward += r
            step += 1
            if info.get('circuit'):
                print(f"Episode {ep} stopped early due to circuit: {info['circuit']}")
        agent.sync_target()
        print(f"Episode {ep} finished. Total PnL: {total_reward:.2f}, Wallet: {env.wallet:.2f}, Epsilon: {agent.epsilon:.4f}")

    # save model optionally
    os.makedirs('models', exist_ok=True)
    agent.save('models/dqn_initial.pth')
    print("Saved model to models/dqn_initial.pth")

if __name__ == '__main__':
    main()
