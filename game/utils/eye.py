"""

This helper module implements the function to confirm eye shape.

"""

import sys

sys.path.append("../")

from go_types import Point


def is_eye(board, point, color):
    # Eye must be empty point.
    if board.get_stone(point) is not None:
        return False

    # All adjacent points must contain friendly stones.
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get_stone(neighbor)

            if neighbor_color != color:
                return False

    friendly_corners  = 0
    off_board_corners = 0

    corners = [
        Point(point.row - 1, point.col - 1),
        Point(point.row - 1, point.col + 1),
        Point(point.row + 1, point.col - 1),
        Point(point.row + 1, point.col + 1),
    ]

    # Eye must control at least three out of four corners if point is in middle of board and all
    # corners if point is on edge or in corner.
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get_stone(corner)

            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1

    # Point is on edge or in corner.
    if off_board_corners > 0:
        return off_board_corners + friendly_corners == 4

    # Point is in middle of board.
    return friendly_corners >= 3
