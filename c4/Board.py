import math
from typing import List

import torch
from torch._C import dtype

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Colors:
    red = '\033[91m'
    yellow = '\033[93m'
    default = '\033[0m'

def add_color(text, color):
    if color == 2:
        return Colors.red + text + Colors.default
    elif color == 1:
        return Colors.yellow + text + Colors.default
    else:
        return text

class Board:
    def __init__(self, width, height, val = None) -> None:
        self.width = width
        self.height = height
        if val:
            self.board = self.from_int(val)
        else:
            self.board = [[0] * width for _ in range(height)]

    def from_int(self, val) -> List:
        board = [[] for _ in range(self.height)]
        for x in range(self.width):
            colval = val % (2 ** (self.height + 1))
            val //= 2 ** (self.height + 1)
            colheight = math.floor(math.log2(colval + 1))
            for y in range(self.height):
                if y < self.height - colheight:
                    board[y].insert(0, 0)
                else:
                    board[y].insert(0, 1 + ((colval - 1) & (2 ** (self.height - 1 - y)) > 0))
        return board

    def available_moves(self) -> list:
        return [i for i in range(self.width) if self.board[0][i] == 0]
    
    def drop_disc(self, id, x) -> int:
        if x < 0 or x >= self.width:
            raise IndexError
        i = self.height - 1
        while self.board[i][x] != 0:
            i -= 1
        self.board[i][x] = id
        return i
    
    def check_win(self, x, y) -> bool:
        return self.count_discs(x, y, 0, 0) >= 4

    def count_discs(self, x, y, v, h) -> int:
        if v == 0 and h == 0:
            return 1 + max([
                self.count_discs(x, y, -1, 0) + self.count_discs(x, y, 1, 0),
                self.count_discs(x, y, 0, -1) + self.count_discs(x, y, 0, 1),
                self.count_discs(x, y, -1, -1) + self.count_discs(x, y, 1, 1),
                self.count_discs(x, y, -1, 1) + self.count_discs(x, y, 1, -1)
            ])
        else:
            try:
                if x + h < 0 or y + v < 0:
                    raise IndexError
                if self.board[y][x] == self.board[y+v][x+h]:
                    return 1 + self.count_discs(x+h, y+v, v, h)
                else:
                    return 0
            except IndexError:
                return 0
    
    def to_int(self) -> int:
        val = 0
        for x in range(self.width):
            val *= 2 ** (self.height + 1)
            xval = 0
            multiplier = 1
            for y in range(self.height):
                state = self.board[self.height - 1 - y][x]
                if state:
                    xval += state * multiplier
                    multiplier *= 2
                else:
                    break
            val += xval
        return val

    def to_tensor(self, pov) -> torch.Tensor:
        t = torch.tensor(self.board, dtype=torch.int64, device=device)
        t = torch.nn.functional.one_hot(t, num_classes=3)
        t = torch.index_select(t, 2, torch.tensor([1, 2], device=device))
        if pov == 2:
            t = torch.flip(t, [2])
        return t.float()
    
    def display(self):
        for row in self.board:
            print('|', end='')
            for val in row:
                if val == 0:
                    print(' ', end='')
                else:
                    print(add_color('@', val), end='')
                print('|', end='')
            print()
