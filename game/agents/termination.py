"""

This module implements the game termination mechanism for the Go agent.

"""

import sys

sys.path.append("../")

from .base import Agent

import go_board_fast
import utils.score


class TerminationStrategy():
    def __init__(self):
        pass

    def should_pass(self, game_state):
        return False

    def should_resign(self, game_state):
        return False

class OpponentPass(TerminationStrategy):
    """
    Go agent should pass when opponent passes.
    """

    def should_pass(self, game_state):
        if game_state.last_move is not None:
            if game_state.last_move.is_pass:
                return True
            else:
                return False

class ResignGG(TerminationStrategy):
    """
    Agent should resign when point discrepancy becomes too large.
    """

    def __init__(self, own_color, cut_off_move, margin):
        self.cut_off_move = cut_off_move
        self.margin       = margin
        self.own_color    = own_color

        self.moves_played = 0

    def should_pass(self, game_state):
        return False

    def should_resign(self, game_state):
        self.moves_played += 1

        if self.moves_played:
            result = utils.score.compute_result(self)

            if result.winner != self.own_color and result.winning_margin >= self.margin:
                return True

        return False

class TerminationAgent(Agent):
    def __init__(self, agent, strategy = None):
        self.agent    = agent
        self.strategy = strategy if strategy is not None else TerminationStrategy()

    def select_move(self, game_state):
        if self.strategy.should_pass(game_state):
            return go_board_fast.Move.pass_turn()
        elif self.strategy.should_resign(game_state):
            return go_board_fast.Move.resign()
        else:
            return self.agent.select_move(game_state)

def return_strategy(strategy):
    if strategy == "opponent_passes":
        return OpponentPass()
    else:
        raise ValueError("Unsupported termination strategy: {0}".format(strategy))
