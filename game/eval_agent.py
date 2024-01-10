"""

This module implements Go agent evaluation using game simulation.

"""

from datetime      import datetime
from eunkyo        import load_agent
from go_board_fast import GameState
from go_board_fast import Player
from utils.play_io import print_board
from utils.score   import compute_result

import argparse
import h5py
import os
import sys


# Disable TensorFlow warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser(
        usage = "python " + sys.argv[0] + " -a agent -o opponent -s 10"
    )

    parser.add_argument(
        "-a", "--agent", required = True, type = str, help = "challenger agent filename prefix"
    )

    parser.add_argument(
        "-b", "--board", default = 9, type = int, help = "Go ban size (default = 9)"
    )

    parser.add_argument(
        "-d", "--disp", action = "store_true", help = "print game results to screen"
    )

    parser.add_argument(
        "-o", "--oppo", required = True, type = str, help = "champion agent filename prefix"
    )

    parser.add_argument(
        "-r", "--rounds", default = 1, type = int,
        help = "number of rounds per move selection (default = 1)"
    )

    parser.add_argument(
        "-s", "--sims", default = 1, type = int,
        help = "number of games to simulate (default = 1)"
    )

    return parser.parse_args()

def simulate_game(agent_black, agent_white, board_size, display = False):
    agents = {
        Player.black: agent_black,
        Player.white: agent_white
    }

    game      = GameState.new_game(board_size)
    num_moves = 0

    while not game.is_over():
        next_move  = agents[game.next_player].select_move(game)
        game       = game.play_move(next_move)
        num_moves += 1

    result = compute_result(game)

    if display:
        print("\n===== Game Result =====\n")
        print_board(game.board)
        print()
        print("Score:", result, "in", num_moves, "moves")
        print()

    return result.winner

def main():
    print("\n========== Agent Evaluation Module ==========\n")

    # Configure command line argument parser.
    args = parse_args()

    board_size = args.board
    display    = args.disp
    rounds     = args.rounds
    sims       = args.sims

    agent_1 = load_agent(h5py.File("./outputs/agent/" + args.agent + ".h5", "r"), rounds)
    agent_2 = load_agent(h5py.File("./outputs/agent/" + args.oppo  + ".h5", "r"), rounds)

    losses = 0
    wins   = 0

    player = Player.black

    print("[+] Running game simulations ...\n")

    eval_start = datetime.now()  # evaluation start time

    for i in range(sims):
        print("[-] Game {0} / {1}".format(i + 1, sims))

        if player == Player.black:
            player_black = agent_1
            player_white = agent_2
        else:
            player_black = agent_2
            player_white = agent_1

        winner = simulate_game(player_black, player_white, board_size, display)

        if winner == player:
            wins += 1
        else:
            losses += 1

        # Swap colors after each game in case either agent plays better with particular color.
        player = player.other

    if not display:
        print()

    print("[+] Total wins: {0} / {1}".format(wins, losses + wins))
    print("\n[+] Evaluation time: {0}".format(datetime.now() - eval_start))

if __name__ == "__main__":
    main()
