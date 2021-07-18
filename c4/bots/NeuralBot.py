from ..Board import Board
from .BasicBot import BasicBot
from torch import nn
import torch

class NeuralBot(BasicBot):

    def __init__(self, width, height):
        super().__init__()
        self.net = DQN(width, height)
    
    def make_move(self, board: Board):
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
