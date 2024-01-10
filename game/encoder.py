"""

This module implements the encoding mechanism to represent the Go board as a tensor. Each element
maps to a point on the board.

"""

from go_board_fast import Move
from go_types      import Player
from go_types      import Point

import numpy as np


# 0 - 3 : player stones with 1, 2, 3 or 4+ liberties
# 4 - 7 : opponent stones with 1, 2, 3 or 4+ liberties
# 8     : 1 if player gets komi
# 9     : 1 if opponent gets komi
# 10    : move illegal due to ko rule
NUM_PLANES = 11

class Encoder():
    def __init__(self, board_size):
        self.board_size = board_size
        self.num_planes = NUM_PLANES
        self.point_move = {}

    def decode_move_index(self, index):
        """
        Board positions are decoded as vector elements.
        """

        if index == self.board_size * self.board_size:
            point = None
        else:
            ro    = index // self.board_size
            co    = index %  self.board_size
            point = Point(row = ro + 1, col = co + 1)

        return self.record_move(point)

    def encode_board(self, game_state):
        board_tensor = np.zeros(self.shape())
        next_player  = game_state.next_player

        if game_state.next_player == Player.white:
            board_tensor[8] = 1
        else:
            board_tensor[9] = 1

        for ro in range(self.board_size):
            for co in range(self.board_size):
                point     = Point(row = ro + 1, col = co + 1)
                go_string = game_state.board.get_string(point)

                if go_string is None:
                    if game_state.ko_rule(next_player, Move.play_stone(point)):
                        board_tensor[10][ro][co] = 1
                else:
                    liberty_plane = min(4, go_string.num_liberties) - 1

                    if go_string.color != next_player:
                        liberty_plane += 4

                    board_tensor[liberty_plane][ro][co] = 1

        return board_tensor

    def encode_move(self, move):
        """
        Moves are represented as vector elements and encoded as board positions.
        Last index represents pass.
        Neural network does not learn resignation.
        """

        if move.is_play:
            return (self.board_size * (move.point.row - 1) + (move.point.col - 1))
        elif move.is_pass:
            return self.board_size * self.board_size

        raise ValueError("Cannot encode resignation!")

    def num_moves(self):
        return self.board_size * self.board_size + 1

    def record_move(self, point):
        if point in self.point_move:
            return self.point_move[point]

        if point is None:
            move = Move.pass_turn()
        else:
            move = Move.play_stone(point)

        self.point_move[point] = move

        return move

    def shape(self):
        return self.num_planes, self.board_size, self.board_size
