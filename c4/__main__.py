from .bots.NeuralBot import NeuralBot
from .bots.BasicBot import BasicBot
import os
import pickle
import sys
from copy import deepcopy
from random import choice, randrange
from .Board import Board
WEIGHT_MAX = 255


# class Colors:
#     red = '\033[91m'
#     yellow = '\033[93m'
#     default = '\033[0m'


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


def get_cols(board, width, height):
    return [[board[height - 1 - y][x] for y in range(height)] for x in range(width)]


def col_to_int(col):
    height = len(col)
    val = 0
    for i in range(height):
        val += col[i] << i
    return val


def col_from_int(val, height):
    if val == 0:
        return [0 for _ in height]
    


def board_to_int(board):
    width = len(board)
    height = len(board[0])
    vert_states = (1 << (height + 1)) - 1
    cols = get_cols(board, width, height)
    state = 0
    for col in cols:
        state *= vert_states
        state += col_to_int(col)
    return state


def rec_tuple(x):
    if type(x) is list:
        return tuple([rec_tuple(y) for y in x])
    else:
        return x


def get_clear_board(width, height):
    return [[0 for _ in range(width)] for _ in range(height)]


def flip_board(board):
    flipped_board = deepcopy(board)
    for row in flipped_board:
        row.reverse()
    return flipped_board


def get_av_moves(board):
    return [col for col in range(len(board[0])) if board[0][col] == 0]


def drop_disc(player, col, state, width, height):
    if col < 0 or col >= width or state[0][col] != 0:
        print('Incorrect column', col)
        raise IndexError
    y = 0
    while y < height - 1 and state[y+1][col] == 0:
        y += 1
    state[y][col] = player
    return check_win(col, y, state)


def check_win(x, y, state):
    return count_discs(x, y, 0, 0, state) >= 4


def count_discs(x, y, v, h, state):
    if v == 0 and h == 0:
        return 1 + max([
            count_discs(x, y, -1, 0, state) + count_discs(x, y, 1, 0, state),
            count_discs(x, y, 0, -1, state) + count_discs(x, y, 0, 1, state),
            count_discs(x, y, -1, -1, state) + count_discs(x, y, 1, 1, state),
            count_discs(x, y, -1, 1, state) + count_discs(x, y, 1, -1, state)
        ])
    else:
        try:
            if x + h < 0 or y + v < 0:
                raise IndexError
            if state[y][x] == state[y+v][x+h]:
                return 1 + count_discs(x+h, y+v, v, h, state)
            else:
                return 0
        except IndexError:
            return 0


# def add_color(text, color):
#     if color == 2:
#         return Colors.red + text + Colors.default
#     elif color == 1:
#         return Colors.yellow + text + Colors.default
#     else:
#         return text


# def display(state):
#     for row in state:
#         print('|', end='')
#         for val in row:
#             if val == 0:
#                 print(' ', end='')
#             else:
#                 print(add_color('@', val), end='')
#             print('|', end='')
#         print()


def two_players(width, height):
    while True:
        game = get_clear_board(width, height)
        won = False
        turn = 0
        while not won and turn < width * height:
            desc = 'Player ' + str((turn & 1) + 1) + '\'s turn.'
            while True:
                try:
                    os.system('cls')
                    display(game)
                    print(desc)
                    col = int(input()) - 1
                    won = drop_disc((turn & 1) + 1, col, game, width, height)
                    break
                except ValueError:
                    desc = 'Please enter a column index.'
                except IndexError:
                    desc = 'You can\'t drop disc in that column.'
            turn += 1
        os.system('cls')
        display(game)
        if won:
            print('Player', (turn - 1 & 1) + 1, 'wins!')
        else:
            print('It\'s a draw!')
        input('Press Enter to play again.')

def move_prompt(id, player, board: Board):
    if player == 0:
        desc = 'Your turn.'
    else:
        desc = 'Player ' + str(player) + ' turn.'
    while True:
        try:
            os.system('cls')
            board.display()
            print(desc)
            x = int(input()) - 1
            y = board.drop_disc(id, x)
            return x, y
        except ValueError:
            desc = 'Please enter a column index.'
        except IndexError:
            desc = 'You can\'t drop disc in that column.'

def two_players(width, height):
    while True:
        board = Board(width, height)
        won = False
        turn = 0
        id = 1
        while not won and turn < width * height:
            x, y = move_prompt(id, id, board)
            won = board.check_win(x, y)
            turn += 1
            id = 3 - id
        os.system('cls')
        board.display()
        if won:
            print('Player', 3 - id, 'wins!')
        else:
            print('It\'s a draw!')
        input('Press Enter to play again.')


def vs_bot(width, height, bot: BasicBot):
    while True:
        board = Board(width, height)
        won = False
        turn = 0
        id = 1
        while not won and turn < width * height:

            x, y = move_prompt(id, 0, board)
            won = board.check_win(x, y)
            turn += 1
            id = 3 - id
            if won or turn >= width * height:
                break

            x = bot.make_move(board)
            y = board.drop_disc(id, x)
            won = board.check_win(x, y)
            turn += 1
            id = 3 - id

        os.system('cls')
        board.display()
        if won:
            if id != 1:
                print('You win!')
                bot.lose()
            else:
                print('You lose.')
                bot.win()
        else:
            print('It\'s a draw!')
            bot.draw()
        input('Press Enter to play again.')


def train(width, height, filename=''):
    if filename:
        try:
            with open(filename, 'rb') as file_in:
                Bot.memory = pickle.load(file_in)
        except FileNotFoundError:
            pass
    games_played = 0
    print('Training...')
    try:
        bot1 = Bot()
        bot2 = Bot()
        while True:
            game = get_clear_board(width, height)
            turn = 0
            while True:
                won = drop_disc(1, bot1.make_move(game), game, width, height)
                turn += 1
                if won or turn >= width * height:
                    break
                won = drop_disc(2, bot2.make_move(game), game, width, height)
                turn += 1
                if won or turn >= width * height:
                    break
            if won:
                if turn & 1:
                    bot1.win()
                    bot2.lose()
                else:
                    bot2.win()
                    bot1.lose()
            else:
                bot1.draw()
                bot2.draw()
            games_played += 1
    except KeyboardInterrupt:
        print("Games played:", games_played)
        if filename:
            with open(filename, 'wb') as file_out:
                pickle.dump(Bot.memory, file_out)
        raise KeyboardInterrupt

def train(width, height, bot1, bot2):
    games_played = 0
    print('Training...')
    try:
        while True:
            board = Board(width, height)
            won = False
            turn = 0
            while not won and turn < width * height:

                x = bot1.make_move(board)
                y = board.drop_disc(id, x)
                won = board.check_win(x, y)
                turn += 1
                if won or turn >= width * height:
                    break

                x = bot2.make_move(board)
                y = board.drop_disc(id, x)
                won = board.check_win(x, y)
                turn += 1

            if won:
                if id != 1:
                    bot1.win()
                    bot2.lose()
                else:
                    bot1.lose()
                    bot2.win()
            else:
                bot1.draw()
                bot2.draw()
            games_played += 1
    except KeyboardInterrupt:
        print(games_played, 'games played.')
        raise KeyboardInterrupt()

def get_args(argv):
    argc = len(argv)
    args = {
        'width': 7,
        'height': 6,
        'opponent': None,
        'filename': '',
        'train': False
    }
    i = 0
    while i < argc:
        if argv[i] == '-r' or argv[i] == '--random':
            if args['opponent']:
                raise RuntimeError
            else:
                args['opponent'] = 'BasicBot'
        elif argv[i] == '-b' or argv[i] == '--bot':
            if args['opponent']:
                raise RuntimeError
            else:
                args['opponent'] = 'NeuralBot'
                i += 1
                if i < argc:
                    args['filename'] = argv[i]
        elif argv[i] == '-t' or argv[i] == '-train':
            if args['opponent']:
                raise RuntimeError
            else:
                args['opponent'] = 'NeuralBot'
                args['train'] = True
                i += 1
                if i < argc:
                    args['filename'] = argv[i]
        i += 1
    if not args['opponent']:
        args['opponent'] = 'player'
    return args

def main(argv):
    try:
        args = get_args(argv)
        os.system('color')
        if args['train']:
            if args['opponent'] == 'NeuralBot':
                train(args['width'], args['height'], args['filename'], NeuralBot(1, args['width'], args['height'], args['filename']), NeuralBot(2))
        else:
            if args['opponent'] == 'NeuralBot':
                vs_bot(args['width'], args['height'], NeuralBot(2, args['width'], args['height'], args['filename']))
            elif args['opponent'] == 'BasicBot':
                vs_bot(args['width'], args['height'], BasicBot())
            elif args['opponent'] == 'player':
                two_players(args['width'], args['height'])

    except KeyboardInterrupt:
        print('Exiting...')
        exit(0)


if __name__ == '__main__':
    main(sys.argv)
