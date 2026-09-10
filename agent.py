"""Linear Q-learning readout on top of a fixed reservoir."""
import numpy as np
from blackjack_env import ACTIONS


def featurize(reservoir_state):
    """Readout features = reservoir state + bias term. Accepts a 1-D state
    or an (n, 1) column vector."""
    state = np.ravel(reservoir_state)
    return np.concatenate([state, [1.0]])


class QReadout:
    def __init__(self, n_features, n_actions=len(ACTIONS), lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=(n_actions, n_features))
        self.lr = lr

    def q_values(self, features):
        return self.W @ features

    def act(self, features, epsilon, valid_actions):
        idxs = [ACTIONS.index(a) for a in valid_actions]
        if np.random.random() < epsilon:
            return ACTIONS[np.random.choice(idxs)]
        q = self.q_values(features)
        best = idxs[np.argmax(q[idxs])]
        return ACTIONS[best]

    def update(self, features, action, reward, next_features, done, gamma=0.0):
        a = ACTIONS.index(action)
        q = self.q_values(features)
        target = reward
        if not done:
            target += gamma * np.max(self.q_values(next_features))
        td_error = target - q[a]
        self.W[a] += self.lr * td_error * features
