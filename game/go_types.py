"""

This module implements the Go player and board intersection objects.

"""

from collections import namedtuple
from enum        import Enum


class Player(Enum):
    black = 0
    white = 1

    @property
    def other(self):
        if self == Player.white:
            return Player.black

        return Player.white

class Point(namedtuple("Point", "row col")):
    def neighbors(self):
        return [
            Point(self.row - 1, self.col),
            Point(self.row + 1, self.col),
            Point(self.row,     self.col - 1),
            Point(self.row,     self.col + 1)
        ]
