"""

This module implements the experience mechanism for reinforcement learning.

"""

import numpy as np


class ExperienceBuffer(object):
    def __init__(self, game_count, states, visit_counts, rewards):
        self.game_count   = game_count
        self.states       = states
        self.visit_counts = visit_counts
        self.rewards      = rewards

    def serialize(self, h5file):
        h5file.create_group("experience")
        h5file.create_group("game")

        h5file["experience"].create_dataset("states",       data = self.states)
        h5file["experience"].create_dataset("visit_counts", data = self.visit_counts)
        h5file["experience"].create_dataset("rewards",      data = self.rewards)

        h5file["game"].attrs["count"] = self.game_count

class ExperienceCollector():
    def __init__(self):
        self._current_episode_states       = []
        self._current_episode_visit_counts = []
        self.states                        = []
        self.visit_counts                  = []
        self.rewards                       = []

    def begin_episode(self):
        self._current_episode_states       = []
        self._current_episode_visit_counts = []

    def complete_episode(self, reward):
        num_states = len(self._current_episode_states)

        self.states       += self._current_episode_states
        self.visit_counts += self._current_episode_visit_counts
        self.rewards      += [reward for _ in range(num_states)]

        # Reset episode buffers.
        self._current_episode_states       = []
        self._current_episode_visit_counts = []

    def record_decision(self, state, visit_counts):
        self._current_episode_states.append(state)
        self._current_episode_visit_counts.append(visit_counts)

def combine_experience(game_count, collectors):
    combined_states       = np.concatenate([np.array(cl.states)       for cl in collectors])
    combined_visit_counts = np.concatenate([np.array(cl.visit_counts) for cl in collectors])
    combined_rewards      = np.concatenate([np.array(cl.rewards)      for cl in collectors])

    return ExperienceBuffer(game_count, combined_states, combined_visit_counts, combined_rewards)

def load_experience(h5file):
    return ExperienceBuffer(
        game_count   = h5file["game"].attrs["count"],
        states       = np.array(h5file["experience"]["states"]),
        visit_counts = np.array(h5file["experience"]["visit_counts"]),
        rewards      = np.array(h5file["experience"]["rewards"])
    )
