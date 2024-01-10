"""

The base module serves as the interface for the Go agent.

"""

class Agent:
    def __init__(self):
        pass

    def select_move(self, game_state):
        raise NotImplementedError()
