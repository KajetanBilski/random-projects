from collections import deque, namedtuple
import pickle
import random
from ..Board import Board
from .BasicBot import BasicBot
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralBot(BasicBot):

    def __init__(self, id):
        super().__init__()
        self.id = id
    
    def make_move(self, board: Board):
        if random.random() > self.eps:
            vals = policy_net(torch.unsqueeze(board.to_tensor(self.id), 0))
            _, indices = vals.squeeze().sort()
            i = 0
            av_moves = board.available_moves()
            while indices[i] not in av_moves:
                i += 1
            return indices[i]
        else:
            return super().make_move(board)

    def win(self):
        return super().win()
    
    def lose(self):
        return super().lose()
    
    def draw(self):
        return super().draw()


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


policy_net = None
target_net = None
criterion = None
optimizer = None
prev_states = None

def init_net(width=7, height=6, filename=None):
    policy_net = DQN(width, height).to(device)
    if filename:
        with open(filename, "rb") as f:
            policy_net.load_state_dict(pickle.load(f))
    target_net = DQN(width, height).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = torch.optim.RMSprop(policy_net.parameters())
    criterion = nn.SmoothL1Loss()

def reset_game():
    prev_states = deque([], maxlen=2)