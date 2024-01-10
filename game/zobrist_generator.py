"""

This module generates Python code to produce Zobrist hashes for recording game states
on 19 x 19 boards.

"""

from go_types import Player
from go_types import Point

import random


MAX_63 = 0x7FFFFFFFFFFFFFFF

table = {}

def python_code(player_state):
    if player_state is None:
        return "None"

    if player_state == Player.black:
        return Player.black

    return Player.white

for row in range(1, 20):
    for col in range(1, 20):
        for state in (None, Player.black, Player.white):
            z_hash = random.randint(0, MAX_63)

            table[Point(row, col), state] = z_hash

# Generate Python code to produce hash.
print("from go_types import Player")
print("from go_types import Point")
print("\n")
print('__all__ = ["EMPTY_BOARD", "Z_HASHES"]')
print()
print("EMPTY_BOARD = %d" % random.randint(0, MAX_63))
print()
print("Z_HASHES = {")

for (point, state), z_hash in table.items():
    print("    (%r, %s): %r," % (point, python_code(state), z_hash))

print("}")
