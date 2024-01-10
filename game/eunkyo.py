"""

This module implements the Eunkyo Go agent and move tree structure.

"""

from agents            import Agent
from encoder           import Encoder
from keras.optimizers  import SGD
from utils.keras_utils import load_model_from_hdf5_group
from utils.keras_utils import save_model_to_hdf5_group

import numpy as np

import h5py
import os


class Branch():
    """
    Each branch represents follow-up move, which may or may not have already been visited, from
    current position.
    """

    def __init__(self, prior):
        self.prior       = prior
        self.visit_count = 0
        self.total_value = 0.0

class TreeNode:
    """
    Each node in game tree represents possible board position.
    """

    def __init__(self, state, value, priors, parent, last_move):
        self.state             = state
        self.value             = value
        self.parent            = parent     # None for root node
        self.last_move         = last_move  # None for root node
        self.total_visit_count = 1

        self.branches = {}
        self.children = {}  # children map move to another node

        for move, prior in priors.items():
            if state.is_valid_move(move):
                self.branches[move] = Branch(prior)

    def add_child(self, move, child_node):
        """
        Add new node to tree.
        """

        self.children[move] = child_node

    def expected_value(self, move):
        """
        Return average over all visits through tree.
        """

        branch = self.branches[move]

        if branch.visit_count == 0:
            return 0.0

        return branch.total_value / branch.visit_count

    def get_child(self, move):
        return self.children[move]

    def has_child(self, move):
        """
        Check if child node exists for move.
        """

        return move in self.children

    def prior(self, move):
        """
        Return prior probability of move (i.e., how good it is expected to be before visiting).
        """

        return self.branches[move].prior

    def record_visit(self, move, value):
        """
        Update tree statistics upon visit.
        """

        self.total_visit_count          += 1
        self.branches[move].visit_count += 1
        self.branches[move].total_value += value

    def valid_moves(self):
        """
        Return list of all possible moves from this node.
        """

        return self.branches.keys()

    def visit_count(self, move):
        """
        Return number of visits to move branch.
        """

        if move in self.branches:
            return self.branches[move].visit_count

        return 0

##################################################
#                    Go Agent                    #
##################################################

class EunkyoAgent(Agent):
    def __init__(self, model, encoder, rounds = 1000, ee = 3.0):
        self.ee         = ee
        self.collector  = None
        self.encoder    = encoder
        self.model      = model
        self.num_rounds = rounds

    def new_node(self, game_state, move = None, parent = None):
        """
        Create new node and add to parent (if exists).
        """

        state_tensor = self.encoder.encode_board(game_state)
        model_input  = np.array([state_tensor])

        # Current version of model.predict() suffers memory leak.
        # priors, values = self.model.predict(model_input, verbose = 0)
        priors, values = self.model(model_input)

        priors = priors[0]
        value  = values[0][0]

        # Add Dirichlet noise, with concentration of 0.05, to root node to introduce randomness in
        # search process. Modify priors as weighted average of true priors and noise.
        if parent is None:
            noise  = np.random.dirichlet(np.ones_like(priors) * 0.05)
            priors = 0.75 * priors + 0.25 * noise

        # Unpack priors vector into dictionary mapping move objects to corresponding
        # prior probabilities.
        move_priors = {
            self.encoder.decode_move_index(i): prior for i, prior in enumerate(priors)
        }

        node = TreeNode(game_state, value, move_priors, parent, move)

        if parent is not None:
            parent.add_child(move, node)

        return node

    def select_branch(self, node):
        """
        Choose branch to traverse according to AlphaGo Zero scoring function based on:
          1) number of times branch has already been visited
          2) estimated value of branch
          3) prior probability of move
        """

        total_visits = node.total_visit_count

        def score_branch(move):
            n = node.visit_count(move)
            p = node.prior(move)
            q = node.expected_value(move)

            return q + self.ee * p * np.sqrt(total_visits) / (n + 1)

        return max(node.valid_moves(), key = score_branch)

    def select_move(self, game_state):
        """
        Traverse game tree to find optimal move to play.
        """

        root = self.new_node(game_state)

        # Each round adds new board position to tree. More rounds per move make tree grow
        # larger (either in breadth or depth) and lead to better moves.
        for _ in range(self.num_rounds):
            node      = root
            next_move = self.select_branch(node)

            # Repeat branch selection until leaf node is reached.
            while node.has_child(next_move):
                node      = node.get_child(next_move)
                next_move = self.select_branch(node)

            # At leaf node, create new node to expand tree.
            new_state  = node.state.play_move(next_move)
            child_node = self.new_node(new_state, parent = node)
            move       = next_move

            # Switch player perspective at each level of tree.
            # Good move for Black means bad move for White and vice versa.
            value = -1 * child_node.value

            # Update statistics of all parent nodes.
            while node is not None:
                node.record_visit(move, value)

                move  = node.last_move
                node  = node.parent
                value = -1 * value

        if self.collector is not None:
            root_state_tensor = self.encoder.encode_board(game_state)

            visit_counts = np.array(
                [root.visit_count(self.encoder.decode_move_index(i))
                 for i in range(self.encoder.num_moves())]
            )

            self.collector.record_decision(root_state_tensor, visit_counts)

        return max(root.valid_moves(), key = root.visit_count)

    def serialize(self, h5file):
        h5file.create_group("encoder")
        h5file.create_group("model")

        h5file["encoder"].attrs["board_size"] = self.encoder.board_size

        save_model_to_hdf5_group(self.model, h5file["model"])

    def set_collector(self, collector):
        self.collector = collector

    def train(self, exp):
        """
        Training target for action output is number of visits made for each move in tree search.
        Training target for value output is 1 if agent won and -1 if agent lost.
        """

        model_input  = exp.states
        num_examples = exp.states.shape[0]
        value_target = exp.rewards

        # Track number of times each move is visited during self-play and normalize.
        visit_sums    = np.sum(exp.visit_counts, axis = 1).reshape((num_examples, 1))
        action_target = exp.visit_counts / visit_sums

        self.model.summary()

        print("\n[+] Training model ...\n")

        self.model.compile(
            loss      = ["categorical_crossentropy", "mse"],
            metrics   = ["accuracy"],
            optimizer = SGD(decay = 0.01, learning_rate = 0.01, momentum = 0.9)
        )

        self.model.fit(
            model_input, [action_target, value_target], batch_size = 512, epochs = 1
        )

        agent_out  = "./outputs/agent/eunkyo"
        agent_out += "_b" + str(self.encoder.board_size)
        agent_out += "_g" + str(exp.game_count + 1)
        agent_out += "_r" + str(self.num_rounds)
        agent_out += ".h5"

        # Save agent to disk.
        if not os.path.exists("./outputs/agent"):
            os.makedirs("./outputs/agent")

        with h5py.File(agent_out, "w") as h5:
            self.serialize(h5)

def load_agent(h5file, rounds = 1000):
    board_size   = h5file["encoder"].attrs["board_size"]
    game_encoder = Encoder(board_size)
    model        = load_model_from_hdf5_group(h5file["model"])

    return EunkyoAgent(model, game_encoder, rounds)
