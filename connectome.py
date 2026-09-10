"""Load the real whole-brain FlyWire connectome as a sparse adjacency
matrix, falling back to a synthetic fly-like matrix if the full download
isn't reachable.

Full data source: https://github.com/eonsystemspbc/fly-brain
(2025_Connectivity_783.parquet: ~138,600 neurons, ~15M signed synapse
pairs, ultimately from FlyWire / Shiu et al. 2023). Downloaded once and
cached next to this file as flywire_full.parquet."""
import os
import urllib.request
import numpy as np
from scipy import sparse

SYNTHETIC_N = 300  # size of the synthetic fallback network
_HERE = os.path.dirname(__file__)
_FULL_CACHE = os.path.join(_HERE, "flywire_full.parquet")
_FULL_URL = (
    "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/"
    "data/2025_Connectivity_783.parquet"
)


def _download_full(path=_FULL_CACHE, url=_FULL_URL, timeout=30):
    if os.path.exists(path):
        return path
    try:
        urllib.request.urlretrieve(url, path)
        return path
    except Exception:
        return None


def _load_whole_brain():
    path = _download_full()
    if path is None:
        return None
    try:
        import pandas as pd
        df = pd.read_parquet(
            path, columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"]
        )
    except Exception:
        return None
    n = int(max(df.Presynaptic_Index.max(), df.Postsynaptic_Index.max())) + 1
    w = (df.Connectivity.values * df.Excitatory.values).astype(np.float32)
    W = sparse.csr_matrix(
        (w, (df.Postsynaptic_Index.values, df.Presynaptic_Index.values)), shape=(n, n)
    )
    return W


def _synthetic_fly_like(n=SYNTHETIC_N, sparsity=0.02, seed=0):
    """Dale's-law sparse matrix with fly-connectome-like statistics, used
    only if the real connectome can't be loaded."""
    rng = np.random.default_rng(seed)
    n_modules = 10
    module_of = rng.integers(0, n_modules, size=n)
    is_exc = rng.random(n) < 0.8
    W = np.zeros((n, n))
    for i in range(n):
        same_module = module_of == module_of[i]
        p = np.where(same_module, sparsity * 5, sparsity)
        p[i] = 0
        connect = rng.random(n) < p
        weights = rng.lognormal(mean=0.0, sigma=0.6, size=n)
        row = connect * weights
        row[~is_exc[i]] *= -1
        W[i] = row
    return sparse.csr_matrix(W)


def load_connectome(seed=0, prefer_full=True):
    """Returns (W, source_label). W is a sparse (n, n) recurrent matrix."""
    if prefer_full:
        W = _load_whole_brain()
        if W is not None:
            return W, "flywire_full_brain"
    return _synthetic_fly_like(seed=seed), "synthetic_fly_like"
