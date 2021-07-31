from collections import deque, namedtuple
import pickle
import random
from ..Board import Board
from .BasicBot import BasicBot
from torch import nn
import torch
import os

MEMORY_CAPACITY = 10000
BATCH_SIZE = 128
GAMMA = 0.999
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
        engine = NeuralEngine(width, height, filename, eps)
    
    def make_move(self, board: Board):
        state = board.to_tensor(self.id)
        if random.random() > engine.eps:
            vals = engine.policy_net(torch.unsqueeze(state, 0))
            _, indices = vals.squeeze().sort()
            i = 0
            av_moves = board.available_moves()
            while indices[i] not in av_moves:
                i += 1
            move = indices[i]
        else:
            move = super().make_move(board)
        engine.add_memory(self.last_state, self.last_action, state, 0.)
        return move

    def win(self):
        engine.add_memory(self.last_state, self.last_action, None, 1.)
        self.reset()
        return super().win()
    
    def lose(self):
        engine.add_memory(self.last_state, self.last_action, None, -1.)
        self.reset()
        return super().lose()
    
    def draw(self):
        engine.add_memory(self.last_state, self.last_action, None, 0.)
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
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 7)
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
        self.eps = EPS_START

        self.policy_net = DQN(width, height).to(device)
        if filename and os.path.exists(filename):
            with open(filename, "rb") as f:
                self.policy_net.load_state_dict(pickle.load(f))
        self.target_net = DQN(width, height).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()

        self.memory = ReplayMemory()
    
    def add_memory(self, state, action, next_state, reward):
        pass

    def save(self):
        if self.filename:
            with open(self.filename, 'wb') as f:
                pickle.dump(self.target_net.state_dict(), f)
        print('Saved.')
