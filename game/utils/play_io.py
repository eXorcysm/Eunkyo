"""

This helper module implements gameplay I/O functions.

"""

import sys

sys.path.append("../")

from go_types import Player
from go_types import Point


COLS = "ABCDEFGHJKLMNOPQRST"

STONE_TO_CHAR = {
    None         : " . ",
    Player.black : " x ",
    Player.white : " o "
}

def coordinates_from_point(point):
    return "%s%d" % (COLS[point.col - 1], point.row)

def point_from_coordinates(coordinates):
    co = COLS.index(coordinates[0]) + 1
    ro = int(coordinates[1:])

    return Point(row = ro, col = co)

def print_board(board):
    for ro in range(board.num_rows, 0, -1):
        line = []

        if ro <= board.num_rows:
            bump = " "
        else:
            bump = ""

        for co in range(1, board.num_cols + 1):
            stone = board.get_stone(Point(row = ro, col = co))

            line.append(STONE_TO_CHAR[stone])

        print("%s%d %s" % (bump, ro, "".join(line)))

    print("    " + "  ".join(COLS[:board.num_cols]))

def print_move(player, move):
    if move.is_pass:
        play = "passes."
    elif move.is_resign:
        play = "resigns!"
    else:
        play = "%s%d" % (COLS[move.point.col - 1], move.point.row)

    print("%s %s" % (player, play))
