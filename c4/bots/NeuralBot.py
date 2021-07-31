from collections import deque, namedtuple
import pickle
import random
from ..Board import Board
from .BasicBot import BasicBot
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
engine = None

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

class NeuralEngine:

    def __init__(self, width, height, filename) -> None:
        self.policy_net = DQN(width, height).to(device)
        if filename:
            with open(filename, "rb") as f:
                self.policy_net.load_state_dict(pickle.load(f))
        self.target_net = DQN(width, height).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters())
        self.criterion = nn.SmoothL1Loss()
