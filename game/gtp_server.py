"""

This is the main script that runs the GTP game server.

"""

from agents import termination
from eunkyo import load_agent
from gtp    import GTPInterface

import h5py


def main():
    print("\n========== Start Game ==========\n")

    model             = h5py.File("./outputs/agent/eunkyo.h5", "r")
    agent             = load_agent(model)
    strategy          = termination.return_strategy("opponent_passes")
    termination_agent = termination.TerminationAgent(agent, strategy)
    game_server       = GTPInterface(termination_agent)

    game_server.run()

if __name__ == "__main__":
    main()
