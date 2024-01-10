"""

This module implements Go agent training using game simulation.

"""

from datetime      import datetime
from encoder       import Encoder
from eunkyo        import EunkyoAgent
from eunkyo        import load_agent
from experience    import ExperienceCollector
from experience    import combine_experience
from experience    import load_experience
from go_board_fast import GameState
from go_board_fast import Player
from utils.play_io import print_board

import networks.nn_medium as nn
import utils.score        as score

import argparse
import h5py
import os
import sys


# Disable TensorFlow warnings.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_game_experience(exp_in):
    print("[+] Loading saved game experience ...\n")

    try:
        return load_experience(h5py.File("./outputs/exp/" + exp_in + ".h5", "r"))
    except FileNotFoundError as error:
        raise error

def parse_args():
    parser = argparse.ArgumentParser(usage = "python " + sys.argv[0] + " -e exp -s 10")

    parser.add_argument(
        "-a", "--agent", type = str, help = "agent filename prefix"
    )

    parser.add_argument(
        "-b", "--board", default = 9, type = int, help = "Go ban size (default = 9)"
    )

    parser.add_argument(
        "-c", "--cont", action = "store_true",
        help = "continue game simulations from experience input"
    )

    parser.add_argument(
        "-d", "--disp", action = "store_true", help = "print game results to screen"
    )

    parser.add_argument(
        "-e", "--exp", type = str, help = "experience input filename prefix"
    )

    parser.add_argument(
        "-r", "--rounds", default = 1, type = int,
        help = "number of rounds per move selection (default = 1)"
    )

    parser.add_argument(
        "-s", "--sims", default = 0, type = int,
        help = "number of games to simulate (default = 0)"
    )

    return parser.parse_args()

def simulate_game(agent_black, agent_white, board_size, collector_black,
                  collector_white, display = False):
    agents = {
        Player.black: agent_black,
        Player.white: agent_white
    }

    collector_black.begin_episode()
    collector_white.begin_episode()

    game      = GameState.new_game(board_size)
    num_moves = 0

    while not game.is_over():
        next_move  = agents[game.next_player].select_move(game)
        game       = game.play_move(next_move)
        num_moves += 1

    result = score.compute_result(game)

    if display:
        print("\n===== Game Result =====\n")
        print_board(game.board)
        print()
        print("Score:", result, "in", num_moves, "moves")
        print()

    # Grant reward to winning agent.
    if result.winner == Player.black:
        collector_black.complete_episode(1)
        collector_white.complete_episode(-1)
    else:
        collector_black.complete_episode(-1)
        collector_white.complete_episode(1)

def main():
    print("\n========== Agent Training Module ==========\n")

    # Configure command line argument parser.
    args = parse_args()

    agent      = args.agent
    board_size = args.board
    cont_sims  = args.cont
    display    = args.disp
    exp_in     = args.exp
    rounds     = args.rounds
    sims       = args.sims

    # Load saved agent from disk or initialize new ones.
    if agent:
        agent_black = load_agent(h5py.File("./outputs/agent/" + agent + ".h5", "r"), rounds)
        agent_white = load_agent(h5py.File("./outputs/agent/" + agent + ".h5", "r"), rounds)
    else:
        game_encoder = Encoder(board_size)
        model        = nn.build_model(game_encoder.shape(), game_encoder.num_moves())

        # Initialize two new game agents with model and game encoder.
        agent_black = EunkyoAgent(model, game_encoder, rounds)
        agent_white = EunkyoAgent(model, game_encoder, rounds)

    # Initialize experience collectors.
    collector_black = ExperienceCollector()
    collector_white = ExperienceCollector()

    # Connect agents to experience collectors.
    agent_black.set_collector(collector_black)
    agent_white.set_collector(collector_white)

    # Run game simulations or train Go agent from experience.
    if sims:
        print("[+] Running game simulations ...\n")

        if cont_sims:
            game_exp  = load_game_experience(exp_in)
            sim_start = game_exp.game_count + 1
        else:
            sim_start = 0

        if sim_start >= sims:
            raise ValueError("Game experience already underwent {0} run(s)!".format(sim_start))

        run_start = datetime.now()  # running start time

        for i in range(sim_start, sims):
            print("[-] Game {0} / {1} --- {2} round(s) per move".format(i + 1, sims, rounds))

            simulate_game(agent_black, agent_white, board_size, collector_black,
                          collector_white, display)

        if not display:
            print()

        print("[+] Running time: {0}".format(datetime.now() - run_start))

        if cont_sims:
            collector_black.states       += game_exp.states.tolist()
            collector_black.visit_counts += game_exp.visit_counts.tolist()
            collector_black.rewards      += game_exp.rewards.tolist()

        # Record simulation experience.
        game_exp = combine_experience(i, [collector_black, collector_white])

        exp_out  = "./outputs/exp/exp"
        exp_out += "_b" + str(board_size)
        exp_out += "_g" + str(sims)
        exp_out += "_r" + str(rounds)
        exp_out += ".h5"

        # Save experience to disk.
        if not os.path.exists("./outputs/exp"):
            os.makedirs("./outputs/exp")

        with h5py.File(exp_out, "w") as h5:
            game_exp.serialize(h5)
    else:
        print("[+] Training agent ...\n")

        game_exp = load_game_experience(exp_in)

        train_start = datetime.now()  # training start time

        agent_black.train(game_exp)

        print("\n[+] Training time: {0}".format(datetime.now() - train_start))

if __name__ == "__main__":
    main()
