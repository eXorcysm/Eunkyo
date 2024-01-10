"""

This module implements the GTP front-end for the Go agent and handles command parsing and
response formatting.

"""

import sys

sys.path.append("../")

from .                 import command
from .                 import response
from agent.termination import TerminationAgent
from .board            import board_to_gtp
from .board            import gtp_to_board
from go_board_fast     import GameState
from go_board_fast     import Move
from utils.play_io     import print_board


HANDICAPS_9  = ["C3", "G7", "C7", "G3", "E5"]
HANDICAPS_13 = ["D4", "K10", "D10", "K4", "G7"]
HANDICAPS_19 = ["D4", "Q16", "D16", "Q4", "D10", "Q10", "K4", "K16", "K10"]

class GTPInterface(object):
    def __init__(self, agent, board_size, termination = None):
        self.agent      = TerminationAgent(agent, termination)
        self.board_size = board_size
        self.game_state = GameState.new_game(board_size)
        self._input     = sys.stdin
        self._output    = sys.stdout
        self._stopped   = False

        self.handlers = {
            "board_size"       : self.handle_board_size,
            "clear_board"      : self.handle_clear_board,
            "fixed_handicap"   : self.handle_fixed_handicap,
            "generate_move"    : self.handle_generate_move,
            "known_command"    : self.handle_known_command,
            "komi"             : self.ignore,
            "show_board"       : self.handle_show_board,
            "time_settings"    : self.ignore,
            "time_left"        : self.ignore,
            "play"             : self.handle_play,
            "protocol_version" : self.handle_protocol_version,
            "quit"             : self.handle_quit,
            "unknown_command"  : self.handle_unknown_command
        }

    def handle_board_size(self, size):
        if int(size) != 9 and int(size) != 13 and int(size) != 19:
            return response.error(
                "Received request for unsupported board size ({0})".format(self.board_size)
            )

        return response.success()

    def handle_clear_board(self):
        self.game_state = GameState.new_game(self.board_size)

        return response.success()

    def handle_fixed_handicap(self, num_stones):
        num_stones = int(num_stones)

        if self.board_size == 9:
            HANDICAPS = HANDICAPS_9
        elif self.board_size == 13:
            HANDICAPS = HANDICAPS_13
        elif self.board_size == 19:
            HANDICAPS = HANDICAPS_19
        else:
            self.handle_board_size(self.board_size)

        max_num_stones = len(HANDICAPS)

        if num_stones > max_num_stones:
            return response.error(
                "Number of handicap stones exceeds limit ({0})".format(max_num_stones)
            )

        for stone in HANDICAPS[:num_stones]:
            self.game_state = self.game_state.play_move(gtp_to_board(stone))

        return response.success()

    def handle_generate_move(self, color):
        move            = self.agent.select_move(self.game_state)
        self.game_state = self.game_state.play_move(move)

        if move.is_pass:
            return response.success("pass")
        elif move.is_resign:
            return response.success("resign")

        return response.success(board_to_gtp(move))

    def handle_known_command(self, command_name):
        return response.bool_to_gtp(command_name in self.handler.keys())

    def handle_play(self, color, move):
        if move.lower() == "pass":
            self.game_state = self.game_state.play_move(Move.pass_turn())
        elif move.lower() == "resign":
            self.game_state = self.game_state.play_move(Move.resign_game())
        else:
            self.game_state = self.game_state.play_move(gtp_to_board(move))

        return response.success()

    def handle_protocol_version(self):
        return response.success("2")

    def handle_quit(self):
        self._stopped = True

        return response.success()

    def handle_show_board(self):
        print_board(self.game_state.board)

        return response.success()

    def handle_time_left(self, color, time, stones):
        ## TODO: Arguments: color color, int time, int stones

        return response.success()

    def handle_time_settings(self, main_time, byo_yomi_time, byo_yomi_stones):
        ## TODO: Arguments: int main_time, int byo_yomi_time, int byo_yomi_stones
        return response.success()

    def handle_unknown_command(self, *args):
        return response.error("Received unrecognized command")

    def ignore(self, *args):
        return response.success()

    def process(self, command):
        handler = self.handlers.get(command.name, self.handle_unknown)

        return handler(*command.args)

    def run(self):
        while not self._stopped:
            input_line = self._input.readline().strip()
            cmd        = command.parse(input_line)
            response   = self.process(cmd)

            self._output.write(response.serialize(cmd, response))
            self._output.flush()
