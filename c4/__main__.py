import os
import sys


class Colors:
    red = '\033[91m'
    yellow = '\033[93m'
    default = '\033[0m'


def main(argv):
    try:
        os.system('color')
        width = 7
        height = 6
        while True:
            game = get_clear_board(width, height)
            won = False
            turn = 1
            while not won:
                turn = 1 - turn
                desc = 'Player ' + str(turn + 1) + '\'s turn.'
                while True:
                    try:
                        os.system('cls')
                        display(game)
                        print(desc)
                        col = int(input()) - 1
                        won = drop_disc(turn + 1, col, game, width, height)
                        break
                    except ValueError:
                        desc = 'Please enter a column index.'
                    except IndexError:
                        desc = 'You can\'t drop disc in that column.'
            os.system('cls')
            display(game)
            print('Player', turn + 1, 'wins!')
            input('Press Enter to play again.')
    except KeyboardInterrupt:
        print('Exiting...')
        exit(0)


def get_clear_board(width, height):
    return [[0 for _ in range(width)] for _ in range(height)]


def drop_disc(player, col, state, width, height):
    if col < 0 or col >= width or state[0][col] != 0:
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
            if state[y][x] == state[y+v][x+h]:
                return 1 + count_discs(x+h, y+v, v, h, state)
            else:
                return 0
        except IndexError:
            return 0


def add_color(text, color):
    if color == 2:
        return Colors.red + text + Colors.default
    elif color == 1:
        return Colors.yellow + text + Colors.default
    else:
        return text


def display(state):
    for row in state:
        print('|', end='')
        for val in row:
            if val == 0:
                print(' ', end='')
            else:
                print(add_color('@', val), end='')
            print('|', end='')
        print()


if __name__ == '__main__':
    main(sys.argv)
