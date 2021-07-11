class Bot:
    memory = {}

    def __init__(self):
        self.move_history = []
        self.losing_strategy = None

    def make_move(self, board):
        tboard = rec_tuple(board)
        fboard = rec_tuple(flip_board(board))
        if tboard in self.memory:
            self.choose_move_from_memory(tboard)
            return self.move_history[-1][1]
        elif fboard in self.memory:
            width = len(board[0])
            self.choose_move_from_memory(fboard)
            return width - 1 - self.move_history[-1][1]
        else:
            self.memory[tboard] = {}
            for move in get_av_moves(tboard):
                self.memory[tboard][move] = 16
            self.choose_move_from_memory(tboard)
            # self.memory[tboard] = get_av_moves(tboard)
            # self.move_history.append((tboard, choice(self.memory[tboard])))
            return self.move_history[-1][1]

    def choose_move_from_memory(self, board):
        s = sum(self.memory[board].values())
        if s:
            r = randrange(s)
            for move in self.memory[board]:
                if r < self.memory[board][move]:
                    self.move_history.append((board, move))
                    return
                else:
                    r -= self.memory[board][move]
        else:
            if self.losing_strategy is None:
                self.losing_strategy = len(self.move_history)
            self.move_history.append((board, choice(list(self.memory[board].keys()))))

    def win(self):
        self.move_history.reverse()
        winning_strat = True
        for move in self.move_history:
            board = self.memory[move[0]]
            if winning_strat and board[move[1]] < WEIGHT_MAX:
                for move2 in board:
                    if move2 == move:
                        board[move2] = WEIGHT_MAX
                    else:
                        board[move2] = 0
                winning_strat = False
            elif board[move[1]] < WEIGHT_MAX:
                board[move[1]] += 1
        self.reset()

    def lose(self):
        self.move_history.reverse()
        losing_strat = True
        for move in self.move_history:
            board = self.memory[move[0]]
            if losing_strat and board[move[1]] > 0:
                board[move[1]] = 0
                losing_strat = False
            elif board[move[1]] > 0:
                board[move[1]] -= 1
        self.reset()

    def draw(self):
        self.reset()

    def reset(self):
        self.move_history = []
        self.losing_strategy = None
