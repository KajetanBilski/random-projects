import sys
from os import system
from random import randrange, choice

FIELDS = 4
COLORS = 6


class Guesser:
    def __init__(self):
        self.av = [int_to_comb(i) for i in range(COLORS ** FIELDS)]

    def guess(self):
        return choice(self.av)

    def check(self, black, white, last_guessed):
        if black == FIELDS:
            return True
        empty = FIELDS - black - white
        self.av.remove(last_guessed)
        new_av = []
        for comb in self.av:
            if count_same(last_guessed, comb) == black and count_diff(last_guessed, comb) == empty:
                new_av.append(comb)
        self.av = new_av
        return False


def int_to_comb(i):
    comb = []
    for field in range(FIELDS):
        comb.append(i % COLORS)
        i //= COLORS
    return comb


def comb_to_int(comb):
    return sum(comb[field] * COLORS ** field for field in range(FIELDS))


def count_same(comb1, comb2):
    count = 0
    for i in range(FIELDS):
        count += comb1[i] == comb2[i]
    return count


def count_diff(comb1, comb2):
    count = 0
    for color in range(COLORS):
        diff = comb1.count(color) - comb2.count(color)
        if diff > 0:
            count += diff
    return count


def display(history, last_guess=None, desc=''):
    system('cls')
    for i in range(1, FIELDS + 1):
        print(i, end=' ')
    print('| b w')
    print('-' * (FIELDS * 2 + 5))
    for row in history:
        for val in row[0:-2]:
            print(val, end=' ')
        print('|', row[-2], row[-1])
    if last_guess:
        for val in last_guess:
            print(val, end=' ')
        print('| - -')
    if desc:
        print(desc)


def guessing():
    while True:
        code = int_to_comb(randrange(COLORS ** FIELDS))
        history = []
        desc = ''
        while True:
            try:
                if history:
                    display(history, None, desc)
                else:
                    system('cls')
                    if desc:
                        print(desc)
                guess = input('Make a guess: ')
                guess = [int(w) for w in guess.split()]
                assert len(guess) == FIELDS
                black = count_same(code, guess)
                white = FIELDS - black - count_diff(code, guess)
                history.append(guess + [black, white])
                desc = ''
                if black == FIELDS:
                    display(history, None, 'You win!')
                    input('Press Enter to continue...')
                    break
            except ValueError:
                desc = 'Please input code numbers.'
            except AssertionError:
                desc = 'Incorrect amount of numbers.'


def main(argv):
    try:
        for arg in argv:
            if arg == '-g':
                guessing()
                return
        while True:
            try:
                guesser = Guesser()
                history = []
                while True:
                    guess = guesser.guess()
                    desc = ''
                    while True:
                        display(history, guess, desc)
                        try:
                            black = int(input('Black: '))
                            white = int(input('White: '))
                            win = guesser.check(black, white, guess)
                            history.append(guess + [black, white])
                            break
                        except ValueError:
                            desc = 'Please enter a number.'
                    if win:
                        print('I win!')
                        input('Press Enter to continue...')
                        break
            except IndexError:
                print('No possible combinations.')
                input('Press Enter to continue...')
    except KeyboardInterrupt:
        print('Exiting...')
        sys.exit(0)


if __name__ == '__main__':
    main(sys.argv)
