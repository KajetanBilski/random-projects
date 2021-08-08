from .bots.NeuralBot import NeuralBot
from .bots.BasicBot import BasicBot
import os
import sys
from copy import deepcopy
from .Board import Board
WEIGHT_MAX = 255


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
                y = board.drop_disc(1, x)
                won = board.check_win(x, y)
                turn += 1
                if won or turn >= width * height:
                    break

                x = bot2.make_move(board)
                y = board.drop_disc(2, x)
                won = board.check_win(x, y)
                turn += 1

            if won:
                if turn % 2:
                    bot1.win()
                    bot2.lose()
                else:
                    bot1.lose()
                    bot2.win()
            else:
                bot1.draw()
                bot2.draw()
            NeuralBot.update_target()
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
                NeuralBot.init_engine(args['width'], args['height'], args['filename'], True)
                train(args['width'], args['height'], NeuralBot(1), NeuralBot(2))
        else:
            if args['opponent'] == 'NeuralBot':
                NeuralBot.init_engine(args['width'], args['height'], args['filename'], False)
                vs_bot(args['width'], args['height'], NeuralBot(2))
            elif args['opponent'] == 'BasicBot':
                vs_bot(args['width'], args['height'], BasicBot())
            elif args['opponent'] == 'player':
                two_players(args['width'], args['height'])

    except KeyboardInterrupt:
        NeuralBot.save()
        print('Exiting...')
        exit(0)


if __name__ == '__main__':
    main(sys.argv)
