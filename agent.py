import numpy as np
from blackjack_env import ACTIONS


def featurize(reservoir_state):
    state = np.ravel(reservoir_state)
    return np.concatenate([state, [1.0]])


class QReadout:
    def __init__(self, n_features, n_actions=len(ACTIONS), lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_actions, n_features))
        self.lr = lr
