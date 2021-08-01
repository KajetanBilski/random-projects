from collections import deque, namedtuple
import pickle
import random
import math

from ..Board import Board
from .BasicBot import BasicBot
from torch import nn
import torch
import os

MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine = None

class NeuralBot(BasicBot):

    def __init__(self, id):
        super().__init__()
        self.id = id
        self.reset()

    @staticmethod
    def init_engine(width=7, height=6, filename='', eps=False):
        global engine
        engine = NeuralEngine(width, height, filename, eps)
    
    @staticmethod
    def save():
        if engine:
            engine.save()
    
    @staticmethod
    def update_target():
        engine.next_episode()
    
    def make_move(self, board: Board):
        state = board.to_tensor(self.id)
        if random.random() > engine.eps:
            vals = engine.policy_net(torch.unsqueeze(state, 0))
            _, indices = vals.squeeze().sort()
            i = 0
            av_moves = board.available_moves()
            while indices[i] not in av_moves:
                engine.push_memory(state.unsqueeze(0), indices[i].unsqueeze(0), None, torch.tensor([-1], device=device))
                i += 1
            action = indices[i]
        else:
            action = torch.tensor(super().make_move(board), device=device)
        if self.last_state is not None:
            engine.push_memory(self.last_state, self.last_action, state.unsqueeze(0), torch.tensor([0], device=device))
        self.last_state = state.unsqueeze(0)
        self.last_action = action.unsqueeze(0)
        return action

    def win(self):
        if self.last_state is not None:
            engine.push_memory(self.last_state, self.last_action, None, torch.tensor([1], device=device))
        self.reset()
        return super().win()
    
    def lose(self):
        if self.last_state is not None:
            engine.push_memory(self.last_state, self.last_action, None, torch.tensor([-1], device=device))
        self.reset()
        return super().lose()
    
    def draw(self):
        if self.last_state is not None:
            engine.push_memory(self.last_state, self.last_action, None, torch.tensor([0], device=device))
        self.reset()
        return super().draw()
    
    def reset(self):
        self.last_state = None
        self.last_action = None


class DQN(nn.Module):

    def __init__(self, width, height):
        if width != 7 or height != 6:
            raise NotImplementedError()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(84, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x: torch.Tensor):
        return self.layers(x)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class NeuralEngine:

    def __init__(self, width, height, filename, eps) -> None:
        
        self.filename = filename
        self.eps_enabled = eps
        if self.eps_enabled:
            self.eps = EPS_START
        else:
            self.eps = 0.
        self.episode = 0
        self.steps = 0

        self.policy_net = DQN(width, height).to(device)
        if filename and os.path.exists(filename):
            with open(filename, "rb") as f:
                self.policy_net.load_state_dict(pickle.load(f))
        self.target_net = DQN(width, height).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayMemory(MEMORY_CAPACITY)
    
    def push_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)
        if self.eps_enabled:
            self.eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps / EPS_DECAY)
        self.steps += 1
        self.optimize_model()
    
    def next_episode(self):
        self.episode += 1
        if self.episode % TARGET_UPDATE:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        if self.filename:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.target_net.state_dict(), f)
        print('Saved.')
