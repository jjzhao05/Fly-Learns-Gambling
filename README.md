# Fly connectome blackjack

Teaching a fly the finest of all human experiences: gambling.

It's actually just a Q-learning layer, which picks hit/stand/double/split, on top of the connectome of a fruit fly.

Watch it play [here](https://jjzhao05-fly-learns-gambling-streamlit-app-fovr1n.streamlit.app/)

## Files
- `connectome.py` - downloads and caches the real connectome
- `reservoir.py` - the frozen fly brain, wrapped as a leaky RNN
- `blackjack_env.py` - 6-deck blackjack rules
- `basic_strategy.py` - textbook strategy table, for comparison
- `agent.py` - the one trained piece, a linear Q-readout
- `train.py` - trains the readout, logs win rate, compares to basic strategy and random
- `policy_report.py` - prints the learned policy vs basic strategy
- `brain_data.py` - real neuron soma positions, for visualizing
- `streamlit_app.py` - live viewer, brain lights up per decision

## Running
```
pip install -r requirements.txt
python train.py
python policy_report.py
```

## Results (138,639 real FlyWire neurons, 10,000 hands)
| Policy | Win rate | EV/hand |
|---|---|---|
| Fly-reservoir Q-agent | 0.385 | -0.161 |
| Basic strategy | 0.418 | -0.059 |
| Random | 0.296 | -0.458 |

Beats random, but does much worse than basic strategy.
