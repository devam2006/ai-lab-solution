"""Warehouse environments and tabular and deep Q-learning experiments."""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

OBSTACLES = {(0, 2), (1, 2), (3, 2), (4, 2)}
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
GAMMA = .95
ROOT = Path(__file__).resolve().parent

def move(p, action):
    dr, dc = MOVES[action]
    q = (p[0] + dr, p[1] + dc)
    invalid = not (0 <= q[0] < 5 and 0 <= q[1] < 5) or q in OBSTACLES
    return (p if invalid else q), invalid

class Warehouse:
    def __init__(self, two=False, start=None):
        self.two = two
        self.start = start or (((2, 0), (2, 4)) if two else (2, 0))
        self.reset()

    def reset(self, state=None):
        self.state = self.start if state is None else state
        self.steps = 0
        return self.state

    def step(self, action):
        self.steps += 1
        if not self.two:
            q, invalid = move(self.state, action)
            terminated = q == (2, 4)
            reward = 10 if terminated else (-1 if invalid else -.1)
            self.state = q
            info = {'invalid': int(invalid), 'collision': False, 'success': terminated}
        else:
            old = self.state
            actions = divmod(action, 5)
            goals = ((2, 4), (2, 0))
            proposals, invalids = [], []
            for p, a, goal in zip(old, actions, goals):
                q, invalid = (p, False) if p == goal else move(p, a)
                proposals.append(q); invalids.append(invalid)
            a, b = proposals
            collision = a == b or (a == old[1] and b == old[0])
            newly = sum(p != g and q == g for p, q, g in zip(old, proposals, goals))
            success = not collision and tuple(proposals) == goals
            terminated = collision or success
            reward = -10 if collision else -.1 - sum(invalids) + 5 * newly + 5 * success
            self.state = old if collision else tuple(proposals)
            info = {'invalid': sum(invalids), 'collision': collision, 'success': success}
        truncated = self.steps >= 40 and not terminated
        return self.state, reward, terminated, truncated, info

def index(p): return 5 * p[0] + p[1]

def encode(state, two=False):
    x = np.zeros(50 if two else 25, dtype=np.float32)
    if two:
        x[index(state[0])] = 1; x[25 + index(state[1])] = 1
    else: x[index(state)] = 1
    return x

def q_update(q, state, action, reward, next_state, terminated, alpha=.2):
    target = reward if terminated else reward + GAMMA * np.max(q[index(next_state)])
    q[index(state), action] += alpha * (target - q[index(state), action])
    return target

def choose(values, epsilon, rng):
    if rng.random() < epsilon: return rng.randrange(len(values))
    best = np.flatnonzero(values == np.max(values))
    return int(rng.choice(best))

def train_table(seed=7, episodes=1500):
    rng = random.Random(seed); env = Warehouse(); q = np.zeros((25, 5))
    records = []
    for ep in range(episodes):
        state = env.reset(); total = 0; invalid = 0
        eps = 1 - .95 * min(ep / 1199, 1)
        while True:
            action = choose(q[index(state)], eps, rng)
            nxt, reward, term, trunc, info = env.step(action)
            q_update(q, state, action, reward, nxt, term)
            total += reward; invalid += info['invalid']; state = nxt
            if term or trunc: break
        records.append((total, int(info['success']), eps, invalid, int(trunc)))
    return q, np.array(records)

class Network(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__(); self.layers = nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))
    def forward(self, x): return self.layers(x)

def targets(rewards, terminated, next_values):
    return rewards + GAMMA * (1 - terminated.float()) * next_values.max(dim=1).values

def train_dqn(two=False, steps=12000, seed=7):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(1)
    rng = random.Random(seed); env = Warehouse(two); n_actions = 25 if two else 5
    online = Network(50 if two else 25, n_actions); target = Network(50 if two else 25, n_actions)
    target.load_state_dict(online.state_dict()); target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=.001); replay = deque(maxlen=20000)
    state = env.reset(); updates = 0; records = []; total = 0; invalid = 0
    for step in range(1, steps + 1):
        eps = 1 - .95 * min((step - 1) / (int(.8 * steps) - 1), 1)
        with torch.no_grad(): values = online(torch.from_numpy(encode(state, two))).numpy()
        action = choose(values, eps, rng)
        nxt, reward, term, trunc, info = env.step(action)
        replay.append((encode(state, two), action, reward, encode(nxt, two), term))
        total += reward; invalid += info['invalid']; state = nxt
        if step >= 500 and step % 4 == 0 and len(replay) >= 64:
            batch = rng.sample(replay, 64)
            states, actions, rewards, next_states, terms = zip(*batch)
            s = torch.from_numpy(np.stack(states)); ns = torch.from_numpy(np.stack(next_states))
            a = torch.tensor(actions, dtype=torch.long); r = torch.tensor(rewards, dtype=torch.float32); d = torch.tensor(terms)
            with torch.no_grad(): y = targets(r, d, target(ns))
            predicted = online(s).gather(1, a[:, None]).squeeze(1)
            loss = nn.functional.mse_loss(predicted, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); updates += 1
            if updates % 250 == 0: target.load_state_dict(online.state_dict())
        if term or trunc:
            records.append((step, total, int(info['success']), int(info['collision']), int(trunc), invalid))
            state = env.reset(); total = 0; invalid = 0
    return online, records

def evaluate(policy, two=False, start=None, random_seed=None):
    env = Warehouse(two, start); state = env.reset(); path = [state]; invalid = 0; total = 0
    rng = random.Random(random_seed)
    for _ in range(40):
        if random_seed is not None: action = rng.randrange(25 if two else 5)
        elif isinstance(policy, np.ndarray): action = int(np.argmax(policy[index(state)]))
        else:
            with torch.no_grad(): action = int(policy(torch.from_numpy(encode(state, two))).argmax().item())
        state, reward, term, trunc, info = env.step(action)
        path.append(state); invalid += info['invalid']; total += reward
        if term or trunc: break
    return {'path': path, 'steps': len(path)-1, 'invalid': invalid, 'success': info['success'], 'collision': info['collision'], 'timeout': trunc, 'return': round(total, 3)}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--two-steps', type=int, default=60000)
    parser.add_argument('--inspect', nargs=3, type=int, metavar=('ROW','COL','ACTION'), help='inspect one single-robot step without training')
    args = parser.parse_args()
    if args.inspect:
        row, col, action = args.inspect
        if (row, col) in OBSTACLES or not (0 <= row < 5 and 0 <= col < 5 and 0 <= action < 5): parser.error('Invalid free cell or action')
        env = Warehouse(); env.reset((row, col)); print(env.step(action)); return
    q, records = train_table(); single, single_log = train_dqn(); double, double_log = train_dqn(True, args.two_steps)
    runs = [evaluate(None, random_seed=i) for i in range(100)]
    result = {'versions': {'numpy': np.__version__, 'torch': torch.__version__}, 'seed': 7,
              'table': evaluate(q), 'table_extra': evaluate(q, start=(0,0)),
              'dqn': evaluate(single), 'dqn_extra': evaluate(single, start=(0,0)),
              'two': evaluate(double, True), 'two_extra': evaluate(double, True, ((0,0),(4,4))),
              'random': {'successes': sum(r['success'] for r in runs), 'timeouts': sum(r['timeout'] for r in runs), 'invalid_total': sum(r['invalid'] for r in runs), 'mean_steps_success': float(np.mean([r['steps'] for r in runs if r['success']])) if any(r['success'] for r in runs) else None}}
    (ROOT/'results.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    np.savetxt(ROOT/'q_training.csv', records, delimiter=',', header='return,success,epsilon,invalid,timeout', comments='')
    np.savetxt(ROOT/'dqn_single_training.csv', single_log, delimiter=',', header='step,return,success,collision,timeout,invalid', comments='', fmt='%g')
    np.savetxt(ROOT/'dqn_two_training.csv', double_log, delimiter=',', header='step,return,success,collision,timeout,invalid', comments='', fmt='%g')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(8,6), sharex=True)
    for i, label in enumerate(['Return', 'Success fraction']):
        ax[i].plot(np.arange(50, len(records)+1), np.convolve(records[:,i], np.ones(50)/50, mode='valid')); ax[i].set_ylabel(label); ax[i].grid(alpha=.3)
    ax[-1].set_xlabel('Episode'); fig.suptitle('Q-learning: 50-episode moving average'); fig.tight_layout(); fig.savefig(ROOT/'q_training.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,6)); ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid()
    for r,c in OBSTACLES: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color='lightgray'))
    for name,color in [('table','tab:blue'),('dqn','tab:orange')]:
        path = result[name]['path']; ax.plot([p[1] for p in path],[p[0] for p in path],marker='o',alpha=.65,label=name,color=color)
        for t,(r,c) in enumerate(path): ax.annotate(str(t),(c,r),xytext=(4,8 if name=='table' else -12),textcoords='offset points',fontsize=7,color=color)
    ax.legend(); ax.set_title('Single-robot greedy rollouts (labels = time)'); fig.tight_layout(); fig.savefig(ROOT/'single_trajectories.png', dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6,6)); ax.set_xlim(-.5,4.5); ax.set_ylim(4.5,-.5); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.grid()
    for r,c in OBSTACLES: ax.add_patch(plt.Rectangle((c-.5,r-.5),1,1,color='lightgray'))
    for robot, color in [(0,'tab:blue'),(1,'tab:orange')]:
        path = [p[robot] for p in result['two']['path']]; ax.plot([p[1] for p in path],[p[0] for p in path],color=color,marker='o',alpha=.65,label='AB'[robot])
        for t,(r,c) in enumerate(path):
            if t and path[t] == path[t-1]: continue
            end = t
            while end + 1 < len(path) and path[end+1] == path[t]: end += 1
            label = str(t) if end == t else f'{t}-{end}'
            ax.annotate(label,(c,r),xytext=(4,8 if robot==0 else -12),textcoords='offset points',fontsize=7,color=color)
    ax.legend(); ax.set_title('Two-robot greedy rollout (labels = time)'); fig.tight_layout(); fig.savefig(ROOT/'two_trajectory.png', dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
