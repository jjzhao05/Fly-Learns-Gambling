import os
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

_HERE = os.path.dirname(__file__)
_FULL_CACHE = os.path.join(_HERE, "flywire_full.parquet")
_FULL_URL = (
    "https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/"
    "data/2025_Connectivity_783.parquet"
)


def _download_full(path=_FULL_CACHE, url=_FULL_URL):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def load_connectome():
    path = _download_full()
    df = pd.read_parquet(
        path, columns=["Presynaptic_Index", "Postsynaptic_Index", "Connectivity", "Excitatory"]
    )
    n = int(max(df.Presynaptic_Index.max(), df.Postsynaptic_Index.max())) + 1
    w = (df.Connectivity.values * df.Excitatory.values).astype(np.float32)
    W = sparse.csr_matrix(
        (w, (df.Postsynaptic_Index.values, df.Presynaptic_Index.values)), shape=(n, n)
    )
    return W, "flywire_full_brain"
