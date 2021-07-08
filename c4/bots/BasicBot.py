from ..Board import Board
from random import choice

class BasicBot:

    def __init__(self):
        pass

    def make_move(self, board: Board):
        av_moves = board.available_moves()
        return choice(av_moves)

    def win(self):
        pass

    def lose(self):
        pass

    def draw(self):
        pass
