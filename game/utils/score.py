"""

This helper module implements functions to compute the game result.

"""

import sys

sys.path.append("../")

from collections import namedtuple
from go_types    import Player
from go_types    import Point


class GameResult(namedtuple("GameResult", "black white komi")):
    def __str__(self):
        white = self.white + self.komi

        if self.black > white:
            return "B+%.1f" % (self.black - white,)

        return "W+%.1f" % (white - self.black,)

    @property
    def winner(self):
        if self.black > self.white + self.komi:
            return Player.black

        return Player.white

    @property
    def winning_margin(self):
        white = self.white + self.komi

        return abs(self.black - white)

class Territory():
    def __init__(self, territory_map):
        self.dame_points = []

        self.num_dame = 0

        self.num_black_stones    = 0
        self.num_black_territory = 0

        self.num_white_stones    = 0
        self.num_white_territory = 0

        for point, status in territory_map.items():
            if status == Player.black:
                self.num_black_stones += 1
            elif status == Player.white:
                self.num_white_stones += 1
            elif status == "territory_black":
                self.num_black_territory += 1
            elif status == "territory_white":
                self.num_white_territory += 1
            elif status == "dame":
                self.num_dame += 1

                self.dame_points.append(point)

# Find contiguous section of board containing point and identify all boundary points.
def _collect_region(start, board, visited = None):
    if visited is None:
        visited = {}

    if start in visited:
        return [], set()

    borders        = set()
    points         = [start]
    visited[start] = True
    here           = board.get_stone(start)
    deltas         = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for ro, co in deltas:
        next = Point(row = start.row + ro, col = start.col + co)

        if not board.is_on_grid(next):
            continue

        neighbor = board.get_stone(next)

        if neighbor == here:
            points, borders = _collect_region(next, board, visited)

            borders |= borders
            points  += points
        else:
            borders.add(neighbor)

    return points, borders

# Evaluate final game state and return result.
def compute_result(game_state):
    territory = evaluate_territory(game_state.board)

    return GameResult(
        territory.num_black_territory + territory.num_black_stones,
        territory.num_white_territory + territory.num_white_stones,
        komi = 5.5
    )

# Map board into territory (points completely surrounded by single color) and dame.
def evaluate_territory(board):
    status = {}

    for ro in range(1, board.num_rows + 1):
        for co in range(1, board.num_cols + 1):
            point = Point(row = ro, col = co)

            if point in status:
                continue

            stone = board.get_stone(point)

            if stone is not None:
                status[point] = board.get_stone(point)
            else:
                group, neighbors = _collect_region(point, board)

                if len(neighbors) == 1:
                    neighbor_stone = neighbors.pop()

                    if neighbor_stone == Player.black:
                        stone_string = "black"
                    else:
                        stone_string = "white"

                    fill_with = "territory_" + stone_string
                else:
                    fill_with = "dame"

                for position in group:
                    status[position] = fill_with

    return Territory(status)
