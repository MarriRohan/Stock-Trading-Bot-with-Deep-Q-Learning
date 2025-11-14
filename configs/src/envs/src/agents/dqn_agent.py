# src/agents/dqn_agent.py
import random
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleQNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q = SimpleQNet(state_dim, action_dim).to(device)
        self.target_q = SimpleQNet(state_dim, action_dim).to(device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.get('learning_rate', 1e-3))
        self.replay = deque(maxlen=cfg.get('replay_size', 50000))
        self.batch_size = cfg.get('batch_size', 64)
        self.gamma = cfg.get('gamma', 0.99)
        self.epsilon = cfg.get('eps_start', 1.0)
        self.eps_min = cfg.get('eps_min', 0.05)
        self.eps_decay = cfg.get('eps_decay', 0.995)
        self.train_start = cfg.get('train_start', 1000)
        self.step_count = 0
        self.target_sync_freq = cfg.get('target_sync_freq', 10)

    def act(self, state):
        """
        state: numpy array
        returns: int action index
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(state_t)
        return int(torch.argmax(qvals, dim=1).item())

    def remember(self, s, a, r, s2, done):
        self.replay.append((np.array(s, copy=True), int(a), float(r), np.array(s2, copy=True), bool(done)))
        self.step_count += 1

    def learn(self):
        if len(self.replay) < max(self.batch_size, self.train_start):
            return None

        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2, done = zip(*batch)
        s = torch.tensor(np.stack(s), dtype=torch.float32, device=device)
        a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1)
        s2 = torch.tensor(np.stack(s2), dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device).unsqueeze(1)

        qvals = self.q(s).gather(1, a)
        with torch.no_grad():
            qnext = self.target_q(s2).max(dim=1)[0].unsqueeze(1)
        target = r + (1.0 - done) * self.gamma * qnext

        loss = nn.functional.mse_loss(qvals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay
            if self.epsilon < self.eps_min:
                self.epsilon = self.eps_min

        # periodically sync target
        if self.step_count % (self.target_sync_freq * self.batch_size) == 0:
            self.sync_target()

        return float(loss.item())

    def sync_target(self):
        self.target_q.load_state_dict(self.q.state_dict())

    def save(self, path):
        torch.save({
            'q_state_dict': self.q.state_dict(),
            'target_state_dict': self.target_q.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=device)
        self.q.load_state_dict(data['q_state_dict'])
        self.target_q.load_state_dict(data.get('target_state_dict', data['q_state_dict']))
        self.optimizer.load_state_dict(data.get('optimizer_state', self.optimizer.state_dict()))
        self.epsilon = data.get('epsilon', self.epsilon)
