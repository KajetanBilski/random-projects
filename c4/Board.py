class Board:
    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.board = [[0] * width for _ in range(height)]

    def available_moves(self) -> list:
        return [i for i in range(self.width) if self.board[0][i] == 0]
    
    def drop_disc(self, id, x) -> int:
        if x < 0 or x >= self.width:
            raise IndexError
        i = self.height - 1
        try:
            while self.board[i][x] != 0:
                i -= 1
            self.board[i][x] = id
        except IndexError:
            pass
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
