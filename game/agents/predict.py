"""

This module implements a deep learning agent for a Go server.

"""

import sys

sys.path.append("../")

from .base     import Agent
from utils.eye import is_eye

import encoder
import go_board_fast
import utils.keras_utils

import numpy as np


class DLAgent(Agent):
    def __init__(self, model, encoder):
        self._encoder = encoder
        self._model   = model

    def select_move(self, game_state):
        num_moves    = self._encoder.num_moves()
        state_tensor = self._encoder.encode_board(game_state)
        model_input  = np.array([state_tensor])

        priors, _ = self._model.predict(model_input, verbose = 0)

        priors = priors[0]

        # Increase distance between most likely and least likely moves.
        priors = priors ** 3

        # Prevent probabilities from being stuck at extremes (0 or 1).
        min_prob = 1e-6
        max_prob = 1 - min_prob
        priors   = np.clip(priors, min_prob, max_prob)

        # Normalize to ensure valid probability distribution.
        priors = priors / np.sum(priors)

        # Convert probabilities into ranked list of candidate moves.
        candidate_moves = np.arange(num_moves)

        # Sample potential candidates.
        ranked_moves = np.random.choice(candidate_moves, num_moves, replace = False, p = priors)

        # Starting from top of ranked list, find first valid move that does not reduce eye space.
        # If no legal and non-self-destructive moves are left, pass turn.
        for i in ranked_moves:
            point = self._encoder.decode_move_index(i)

            if game_state.is_valid_move(go_board_fast.Move.play_stone(point)) and not \
               is_eye(game_state.board, point, game_state.next_player):
                return go_board_fast.Move.play_stone(point)

            return go_board_fast.Move.pass_turn()

    def serialize(self, h5file):
        h5file.create_group("encoder")
        h5file.create_group("model")

        h5file["encoder"].attrs["board_size"] = self._encoder.board_size

        utils.keras_utils.save_model_to_hdf5_group(self._model, h5file["model"])

def load_predict_agent(h5file):
    board_size   = h5file["encoder"].attrs["board_size"]
    game_encoder = encoder.Encoder(board_size)
    model        = utils.keras_utils.load_model_from_hdf5_group(h5file["model"])

    return DLAgent(model, game_encoder)
