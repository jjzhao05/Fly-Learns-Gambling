import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs


def _spectral_radius(W, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    try:
        val = eigs(W, k=1, return_eigenvectors=False, maxiter=5000)
        return abs(val[0])
    except Exception:
        pass
    v = rng.normal(size=n)
    v /= np.linalg.norm(v)
    lam = 1.0
    for _ in range(iters):
        v2 = W @ v
        lam = np.linalg.norm(v2)
        if lam == 0:
            return 1.0
        v = v2 / lam
    return lam


class Reservoir:
    def __init__(self, W, n_inputs, spectral_radius=0.9, leak=0.3, seed=0):
        rng = np.random.default_rng(seed)
        n = W.shape[0]
        radius = _spectral_radius(W, seed=seed)
        self.W = sparse.csr_matrix(W) * (spectral_radius / radius)
        self.W_in = rng.uniform(-1, 1, size=(n, n_inputs)).astype(np.float32)
        self.leak = leak
        self.n = n

    def reset(self, batch=1):
        self.state = np.zeros((self.n, batch), dtype=np.float32)
        return self.state

    def step(self, x):
        pre = self.W @ self.state + self.W_in @ x
        self.state = (1 - self.leak) * self.state + self.leak * np.tanh(pre)
        return self.state
