import os
import urllib.request
import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

_HERE = os.path.dirname(__file__)
PARQUET = os.path.join(_HERE, "flywire_full.parquet")
ANNOTATIONS = os.path.join(_HERE, "flywire_annotations.tsv")
ANNOTATIONS_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
CACHE = os.path.join(_HERE, "positions_full.npz")
CATS = ["optic", "central", "sensory", "other"]


def full_brain_positions(n):
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        if len(d["pos_x"]) == n:
            return d["pos_x"], d["pos_y"], d["category"]

    df = pd.read_parquet(PARQUET, columns=[
        "Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index"])
    id_to_index = pd.concat([
        df[["Presynaptic_ID", "Presynaptic_Index"]].rename(columns={"Presynaptic_ID": "id", "Presynaptic_Index": "idx"}),
        df[["Postsynaptic_ID", "Postsynaptic_Index"]].rename(columns={"Postsynaptic_ID": "id", "Postsynaptic_Index": "idx"}),
    ]).drop_duplicates("idx").set_index("idx")["id"]

    if not os.path.exists(ANNOTATIONS):
        urllib.request.urlretrieve(ANNOTATIONS_URL, ANNOTATIONS)
    ann = pd.read_csv(ANNOTATIONS, sep="\t", usecols=["root_id", "pos_x", "pos_y", "super_class"])
    ann = ann.dropna(subset=["pos_x", "pos_y"]).drop_duplicates("root_id").set_index("root_id")

    ids = id_to_index.reindex(range(n))
    joined = ann.reindex(ids.values)
    pos_x = joined["pos_x"].to_numpy(dtype=np.float32)
    pos_y = joined["pos_y"].to_numpy(dtype=np.float32)
    super_class = joined["super_class"].fillna("unknown").to_numpy()
    missing = np.isnan(pos_x)
    pos_x[missing] = np.nanmedian(pos_x)
    pos_y[missing] = np.nanmedian(pos_y)
    cat_code = np.array(
        [CATS.index(c) if c in ("optic", "central", "sensory") else CATS.index("other") for c in super_class],
        dtype=np.int8)

    np.savez(CACHE, pos_x=pos_x, pos_y=pos_y, category=cat_code)
    return pos_x, pos_y, cat_code


def spectral_positions(W):
    Wd = abs(W)
    u, s, _ = svds(Wd + Wd.T, k=2)
    emb = u * s
    return emb[:, 0].astype(np.float32), emb[:, 1].astype(np.float32)
