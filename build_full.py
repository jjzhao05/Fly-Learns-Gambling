import os
import sys
import time
import json
import urllib.request
import numpy as np
import pandas as pd
from scipy import sparse

T0 = time.time()
_HERE = os.path.dirname(__file__)
PARQUET = os.path.join(_HERE, "flywire_full.parquet")
ANNOTATIONS = os.path.join(_HERE, "flywire_annotations.tsv")
ANNOTATIONS_URL = ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                    "main/supplemental_files/Supplemental_file1_neuron_annotations.tsv")
TOPK = 2000
TRAIN_HANDS = 8000
BATCH = 64
N_TRACE_HANDS = 20

if not os.path.exists(ANNOTATIONS):
    urllib.request.urlretrieve(ANNOTATIONS_URL, ANNOTATIONS)

df = pd.read_parquet(PARQUET, columns=[
    "Presynaptic_ID", "Postsynaptic_ID", "Presynaptic_Index", "Postsynaptic_Index",
    "Connectivity", "Excitatory"])
n = int(max(df.Presynaptic_Index.max(), df.Postsynaptic_Index.max())) + 1
w = (df.Connectivity.values * df.Excitatory.values).astype(np.float32)
W = sparse.csr_matrix((w, (df.Postsynaptic_Index.values, df.Presynaptic_Index.values)), shape=(n, n))
print(f"[{time.time()-T0:.0f}s] loaded connectome: n={n} nnz={W.nnz}")

id_to_index = pd.concat([
    df[["Presynaptic_ID", "Presynaptic_Index"]].rename(columns={"Presynaptic_ID": "id", "Presynaptic_Index": "idx"}),
    df[["Postsynaptic_ID", "Postsynaptic_Index"]].rename(columns={"Postsynaptic_ID": "id", "Postsynaptic_Index": "idx"}),
]).drop_duplicates("idx").set_index("idx")["id"]
del df

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
super_class[missing] = "unknown"
print(f"[{time.time()-T0:.0f}s] joined positions, missing={missing.sum()}/{n}")

CATS = ["optic", "central", "sensory", "visual_projection", "ascending", "descending",
        "sensory_ascending", "visual_centrifugal", "motor", "endocrine", "unknown"]
cat_code = np.array([CATS.index(c) if c in CATS else CATS.index("unknown") for c in super_class], dtype=np.int8)

rng = np.random.default_rng(0)
nnz = W.nnz
sample_n = min(18000, nnz)
sample_idx = rng.choice(nnz, size=sample_n, replace=False)
Wc = W.tocoo()
edge_sample = np.stack([Wc.row[sample_idx], Wc.col[sample_idx]], axis=1).astype(np.int32)

sys.path.insert(0, _HERE)
from reservoir import Reservoir
from agent import QReadout, featurize
from blackjack_env import BlackjackEnv, ACTIONS
from train import play_batch, scaled_lr, epsilon_at, raw_input_vec

reservoir = Reservoir(W, n_inputs=3, seed=0)
print(f"[{time.time()-T0:.0f}s] reservoir spectral radius rescaled")
n_features = n + 1
agent = QReadout(n_features=n_features, lr=scaled_lr(n_features), seed=0)

hands_done = 0
for b in range(TRAIN_HANDS // BATCH):
    eps = epsilon_at(hands_done)
    envs = [BlackjackEnv(seed=1000 * b + i) for i in range(BATCH)]
    play_batch(envs, reservoir, agent, eps, learn=True)
    hands_done += BATCH
    if b % 20 == 0:
        print(f"[{time.time()-T0:.0f}s] trained {hands_done} hands")
print(f"[{time.time()-T0:.0f}s] training done: {hands_done} hands")

trace = []
env = BlackjackEnv(seed=999)
for h in range(N_TRACE_HANDS):
    obs = env.reset()
    hand = {
        "initial_player": list(env.player), "initial_dealer_up": env.dealer[0],
        "final_dealer": list(env.dealer), "steps": [],
        "natural": env.done, "reward": env.natural_reward if env.done else None,
    }
    if not env.done:
        reservoir.reset(batch=1)
        first_action = True
        total, usable, dealer_up = obs
        while True:
            x = raw_input_vec(total, usable, dealer_up).reshape(3, 1)
            state = reservoir.step(x)
            features = featurize(state)
            valid = ACTIONS if first_action else ("hit", "stand")
            q = agent.W @ features
            idxs = [ACTIONS.index(a) for a in valid]
            action = ACTIONS[idxs[int(np.argmax(q[idxs]))]]
            obs, reward, done = env.step(action)
            flat = state.ravel()
            top = np.argsort(-np.abs(flat))[:TOPK]
            hand["steps"].append({
                "total": total, "usable": usable, "dealer_up": dealer_up, "action": action,
                "drawn_card": env.player[-1] if action in ("hit", "double") else None,
                "active": [[int(i), round(float(flat[i]), 3)] for i in top],
            })
            if done:
                hand["reward"] = reward
                hand["final_dealer"] = list(env.dealer)
                break
            total, usable, dealer_up = obs
            first_action = False
    trace.append(hand)
    print(f"[{time.time()-T0:.0f}s] traced hand {h+1}/{N_TRACE_HANDS}")

data = {
    "n_neurons": n,
    "positions": np.stack([pos_x, pos_y], axis=1).round(0).astype(int).tolist(),
    "category": cat_code.tolist(),
    "categories": CATS,
    "edges": edge_sample.tolist(),
    "hands": trace,
}
with open("data_full.json", "w") as f:
    json.dump(data, f)
print(f"[{time.time()-T0:.0f}s] wrote data_full.json")
