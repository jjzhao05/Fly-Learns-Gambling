# Fly connectome blackjack

A fixed (untrained) fly-connectome-like recurrent reservoir feeds hand
total, usable-ace flag, and dealer upcard into a network of 300 neurons.
Only a linear readout on top of the reservoir is trained, via one-step
Q-learning, to pick hit/stand/double.

## Files
- `connectome.py` - downloads the full real FlyWire connectome (~138,600
  neurons, ~15M signed synapse pairs) from
  [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)'s
  `2025_Connectivity_783.parquet` (ultimately sourced from FlyWire / Shiu
  et al. 2023), caches it as `flywire_full.parquet`, and builds it as a
  sparse matrix (`scipy.sparse`). If the download fails it falls back to
  `flywire_subgraph_300.csv`, a bundled real 300-neuron induced subgraph
  (the 300 highest-degree neurons from the same dataset, kept densely
  connected). If even that's missing, it falls back to a synthetic sparse
  matrix matched to fly-connectome statistics (Dale's law, ~80/20 exc/inh,
  log-normal weights, modular wiring). Reports whichever source it used:
  `flywire_full_brain`, `flywire_real_subsample`, or `synthetic_fly_like`.
- `reservoir.py` - wraps the matrix as a leaky-integrator RNN (spectral
  radius rescaled to 0.9 via sparse power iteration, fixed random input
  projection), and supports batched state so many hands can be run through
  the reservoir in one sparse matrix multiply. Never trained.
- `blackjack_env.py` - 6-deck shoe, dealer stands on all 17s, 3:2 naturals,
  double down on the first two cards.
- `basic_strategy.py` - published basic strategy table (hit/stand/double).
- `agent.py` - the only trained piece: a linear readout (3 actions x
  n_neurons+1 features) trained by one-step Q-learning on win/loss/push
  reward. Its learning rate is scaled down as the reservoir grows (see
  `scaled_lr` in `train.py`), since the raw per-sample SGD step otherwise
  grows with the feature vector's dimension and diverges on a 138k-neuron
  reservoir.
- `train.py` - simulates hands in batches (default 64 at once) so the
  reservoir's sparse matrix-vector product is amortized across many hands
  per step, logs win rate, evaluates the greedy policy, and compares it
  to basic strategy and random play.
- `policy_report.py` - prints/saves the learned policy grid next to basic
  strategy so disagreements are visible at a glance.
- `results_training_curve.png`, `results_report.txt`, `results_policy_grid.txt`
  - output of the last training run.

## Running
```
pip install -r requirements.txt
python train.py
python policy_report.py
```

## Results (whole-brain reservoir: 138,639 real FlyWire neurons)
10,000 training hands (the sparse matrix-vector product per reservoir
step costs ~14ms per hand at this size, ~500x the 300-neuron reservoir,
so far fewer hands fit in a reasonable run):

| Policy | Win rate | EV/hand |
|---|---|---|
| Fly-reservoir Q-agent | 0.385 | -0.161 |
| Basic strategy (same env) | 0.418 | -0.059 |
| Random | 0.296 | -0.458 |

Still clearly above random with zero hand-coded rules, but further from
basic strategy than the 300-neuron reservoir got after 300k hands (see
git history / `results_policy_grid.txt`) - it's undertrained, not a sign
the extra neurons hurt. `results_policy_grid.txt` shows the correct
hit/stand shape emerging for 17+ and clean 12-16-vs-weak-dealer-card
decisions, but noisier decisions on low hard totals (4-11) where more
training would resolve the "always hit" pattern basic strategy uses.
Doubling is still never learned, for the same reason as before: one-step
TD rarely reinforces a rare, higher-variance action.

To use a smaller, faster-training reservoir instead of the whole brain,
delete `flywire_full.parquet` (if present) and call
`load_connectome(prefer_full=False)` in `train.py`, which uses the bundled
300-neuron real subgraph.
