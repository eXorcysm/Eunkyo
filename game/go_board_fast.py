"""

This updated module implements game mechanics, which include the game board, game state, player
moves and stone string objects. Game states are stored as Zobrist hashes.

"""

from copy        import copy
from copy        import deepcopy
from go_types    import Player
from go_types    import Point
from utils.score import compute_result

import zobrist

import numpy as np


corner_tables   = {}
neighbor_tables = {}

class GameState():
    def __init__(self, board, next_player, previous, move):
        self.board          = board
        self.last_move      = move
        self.next_player    = next_player
        self.previous_state = previous

        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous.previous_states | {(previous.next_player, previous.board.zobrist_hash())}
            )

    def is_over(self):
        if self.last_move is None:
            return False

        if self.last_move.is_resign:
            return True

        second_last_move = self.previous_state.last_move

        if second_last_move is None:
            return False

        return self.last_move.is_pass and second_last_move.is_pass

    def is_self_capture(self, player, move):
        if not move.is_play:
            return False

        return self.board.is_self_capture(player, move.point)

    def is_valid_move(self, move):
        if self.is_over():
            return False

        if move.is_pass or move.is_resign:
            return True

        return (
            self.board.get_stone(move.point) is None         and \
            not self.is_self_capture(self.next_player, move) and \
            not self.ko_rule(self.next_player, move)
        )

    def ko_rule(self, player, move):
        if not move.is_play:
            return False

        if not self.board.will_capture(player, move.point):
            return False

        next_board = deepcopy(self.board)

        next_board.place_stone(player, move.point)

        next_situation = (player.other, next_board.zobrist_hash())

        return next_situation in self.previous_states

    def legal_moves(self):
        moves = []

        for row in range(1, self.board.num_rows + 1):
            for col in range(1, self.board.num_cols + 1):
                move = Move.play_stone(Point(row, col))

                if self.is_valid_move(move):
                    moves.append(move)

        # These moves are always legal.
        moves.append(Move.pass_turn())
        moves.append(Move.resign_game())

        return moves

    @classmethod
    def new_game(cls, board_size):
        if isinstance(board_size, int):
            board_size = (board_size, board_size)

        board = GoBoard(*board_size)

        return GameState(board, Player.black, None, None)

    def play_move(self, move):
        """
        Return new GameState after playing move.
        """

        if move.is_play:
            next_board = deepcopy(self.board)

            next_board.place_stone(self.next_player, move.point)
        else:
            next_board = self.board

        return GameState(next_board, self.next_player.other, self, move)

    @property
    def situation(self):
        return (self.next_player, self.board)

    def winner(self):
        if not self.is_over():
            return None

        if self.last_move.is_resign:
            return self.next_player

        result = compute_result(self)

        return result.winner

class GoBoard():
    def __init__(self, num_rows, num_cols):
        self._grid = {}
        self._hash = zobrist.EMPTY_BOARD

        self.num_cols = num_cols
        self.num_rows = num_rows

        global neighbor_tables

        dim = (num_rows, num_cols)

        if dim not in corner_tables:
            init_corner_table(dim)

        if dim not in neighbor_tables:
            init_neighbor_table(dim)

        self.corner_table   = corner_tables[dim]
        self.neighbor_table = neighbor_tables[dim]

        self.move_ages = MoveAge(self)

    def __deepcopy__(self, memory = {}):
        board_copy = GoBoard(self.num_rows, self.num_cols)

        # Shallow copy is possible because dictionary maps tuples (immutable)
        # to GoString (also immutable).
        board_copy._grid = copy(self._grid)
        board_copy._hash = self._hash

        return board_copy

    def __equal__(self, other):
        return isinstance(other, GoBoard)   and \
            self.num_rows == other.num_rows and \
            self.num_cols == other.num_cols and \
            self._hash()  == other._hash()

    def _remove_string(self, go_string):
        for point in go_string.stones:
            self.move_ages.reset_age(point)

            # Removing string can create liberties for other strings.
            for neighbor in self.neighbor_table[point]:
                neighbor_string = self._grid.get(neighbor)

                if neighbor_string is None:
                    continue

                if neighbor_string is not go_string:
                    self._replace_string(neighbor_string.with_liberty(point))

            self._grid[point] = None

            # Undo Zobrist hashing for removal.
            self._hash ^= zobrist.Z_HASHES[point, go_string.color]

            # Replace with empty point hash.
            self._hash ^= zobrist.Z_HASHES[point, None]

    def _replace_string(self, go_string):
        for point in go_string.stones:
            self._grid[point] = go_string

    def corners(self, point):
        return self.corner_table[point]

    def get_stone(self, point):
        """
        Return content of board point (None if empty or player if stone fills point).
        """

        go_string = self._grid.get(point)

        if go_string is None:
            return None

        return go_string.color

    def get_string(self, point):
        """
        Return entire stone string at board point (None if empty or string if stone fills point).
        """

        go_string = self._grid.get(point)

        if go_string is None:
            return None

        return go_string

    def is_on_grid(self, point):
        if (1 <= point.row <= self.num_rows and 1 <= point.col <= self.num_cols):
            return True

        return False

    def is_self_capture(self, player, point):
        friendly_strings = []

        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)

            if neighbor_string is None:
                return False
            elif neighbor_string.color == player:
                friendly_strings.append(neighbor_string)
            else:
                if neighbor_string.num_liberties == 1:
                    return False

        if all(neighbor.num_liberties == 1 for neighbor in friendly_strings):
            return True

        return False

    def place_stone(self, player, point):
        assert(self.is_on_grid(point))

        if self._grid.get(point) is not None:
            print("Illegal play on %s!" % str(point))

        assert(self._grid.get(point) is None)

        adjacent_same_color     = []
        adjacent_opposite_color = []
        liberties               = []

        self.move_ages.increment_all()
        self.move_ages.add_age(point)

        # Inspect neighboring points for liberties.
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)

            if neighbor_string is None:
                liberties.append(neighbor)
            elif neighbor_string.color == player:
                if neighbor_string not in adjacent_same_color:
                    adjacent_same_color.append(neighbor_string)
            else:
                if neighbor_string not in adjacent_opposite_color:
                    adjacent_opposite_color.append(neighbor_string)

        go_string = GoString(player, [point], liberties)

        # Merge any adjacent stones of same color.
        for same_color_string in adjacent_same_color:
            go_string = go_string.merged_with(same_color_string)

        # Register new stone string to update game board.
        for new_string_point in go_string.stones:
            self._grid[new_string_point] = go_string

        # Remove empty point hash.
        self._hash ^= zobrist.Z_HASHES[point, None]

        # Apply Zobrist hashing for point and player.
        self._hash ^= zobrist.Z_HASHES[point, player]

        # Reduce liberties of adjacent stones of opposite color.
        # Remove stones of opposite color with zero liberties.
        for other_color_string in adjacent_opposite_color:
            replacement = other_color_string.without_liberty(point)

            if replacement.num_liberties:
                self._replace_string(other_color_string.without_liberty(point))
            else:
                self._remove_string(other_color_string)

    def neighbors(self, point):
        return self.neighbor_table[point]

    def will_capture(self, player, point):
        for neighbor in self.neighbor_table[point]:
            neighbor_string = self._grid.get(neighbor)

            if neighbor_string is None:
                continue
            elif neighbor_string.color == player:
                continue
            else:
                if neighbor_string.num_liberties == 1:
                    return True

        return False

    def zobrist_hash(self):
        return self._hash

class GoString():
    """
    Stone strings are chains of connected stones of same color.
    """

    def __init__(self, color, stones, liberties):
        self.color     = color
        self.liberties = frozenset(liberties)
        self.stones    = frozenset(stones)

    def __deepcopy__(self, memory = {}):
        return GoString(self.color, self.stones, deepcopy(self.liberties))

    def __equal__(self, other):
        return isinstance(other, GoString)    and \
            self.color     == other.color     and \
            self.liberties == other.liberties and \
            self.stones    == other.stones

    def merged_with(self, go_string):
        assert(go_string.color == self.color)

        combined_stones = self.stones | go_string.stones

        return GoString(
            self.color,
            combined_stones,
            (self.liberties | go_string.liberties) - combined_stones
        )

    @property
    def num_liberties(self):
        return len(self.liberties)

    def without_liberty(self, point):
        new_liberties = self.liberties - set([point])

        return GoString(self.color, self.stones, new_liberties)

    def with_liberty(self, point):
        new_liberties = self.liberties | set([point])

        return GoString(self.color, self.stones, new_liberties)

class Move():
    def __init__(self, point = None, is_pass = False, is_resign = False):
        assert((point is not None) ^ is_pass ^ is_resign)  # only one can be true

        self.point     = point
        self.is_pass   = is_pass
        self.is_play   = (self.point is not None)
        self.is_resign = is_resign

    def  __equal__(self, other):
        move_a = (self.point, self.is_pass, self.is_play, self.is_resign)
        move_b = (other.point, other.is_pass, other.is_play, other.is_resign)

        return (move_a == move_b)

    def __hash__(self):
        return hash((self.point, self.is_pass, self.is_play, self.is_resign))

    def __str__(self):
        if self.is_pass:
            return "pass"

        if self.is_resign:
            return "resign"

        return "(Row %d, Col %d)" % (self.point.row, self.point.col)

    @classmethod
    def pass_turn(cls):
        return Move(is_pass = True)

    @classmethod
    def play_stone(cls, play_point):
        return Move(point = play_point)

    @classmethod
    def resign_game(cls):
        return Move(is_resign = True)

class MoveAge():
    def __init__(self, board):
        self.move_ages = -np.ones((board.num_rows, board.num_cols))

    def add_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = 0

    def get_age(self, row, col):
        return self.move_ages[row, col]

    def increment_all(self):
        self.move_ages[self.move_ages > -1] += 1

    def reset_age(self, point):
        self.move_ages[point.row - 1, point.col - 1] = -1

def init_corner_table(dim):
    rows, cols = dim

    new_table = {}

    for ro in range(1, rows + 1):
        for co in range(1, cols + 1):
            point = Point(row = ro, col = co)

            full_corners = [
                Point(row = point.row - 1, col = point.col - 1),
                Point(row = point.row - 1, col = point.col + 1),
                Point(row = point.row + 1, col = point.col - 1),
                Point(row = point.row + 1, col = point.col + 1)
            ]

            true_corners = [
                neighbor for neighbor in full_corners
                if 1 <= neighbor.row <= rows and 1 <= neighbor.col <= cols
            ]

            new_table[point] = true_corners

    corner_tables[dim] = new_table

def init_neighbor_table(dim):
    rows, cols = dim

    new_table = {}

    for ro in range(1, rows + 1):
        for co in range(1, cols + 1):
            point          = Point(row = ro, col = co)
            full_neighbors = point.neighbors()

            true_neighbors = [
                neighbor for neighbor in full_neighbors
                if 1 <= neighbor.row <= rows and 1 <= neighbor.col <= cols
            ]

            new_table[point] = true_neighbors

    neighbor_tables[dim] = new_table
