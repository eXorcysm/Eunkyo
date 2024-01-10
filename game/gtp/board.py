"""

This module implements functions to convert GTP coordinates to (row, col) tuple and vice versa.

"""

import sys

sys.path.append("../")

from go_board_fast import Move
from go_types      import Point


COLS = "ABCDEFGHJKLMNOPQRST"

def board_to_gtp(move):
    """
    Convert (row, col) tuple to GTP board coordinates [e.g., (1, 1) => "A1"].
    """

    point = move.point

    return COLS[point.col - 1] + str(point.row)

def gtp_to_board(gtp):
    """
    Convert GTP board coordinates to (row, col) tuple [e.g., "A1" => (1, 1)].
    """

    point = Point(int(gtp[1:]), COLS.find(gtp[0].upper()) + 1)

    return Move(point)
