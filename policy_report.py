"""Compare the trained agent's learned policy grid to basic strategy."""
import numpy as np
from connectome import load_connectome
from reservoir import Reservoir
from blackjack_env import ACTIONS
from agent import QReadout, featurize
from basic_strategy import basic_strategy_action
from train import raw_input_vec


def build_policy_grid(agent, reservoir):
    lines = []
    header = "     " + "".join(f"{u:>4}" for u in range(2, 12))
    for usable in (False, True):
        label = "SOFT" if usable else "HARD"
        lines.append(f"\n{label} totals (rows) vs dealer upcard 2-11 (cols)")
        totals = range(13, 22) if usable else range(4, 22)
        lines.append(header)
        for total in totals:
            row = [f"{total:>4} "]
            for dealer_up in range(2, 12):
                reservoir.reset(batch=1)
                x = raw_input_vec(total, usable, dealer_up).reshape(3, 1)
                state = reservoir.step(x)
                features = featurize(state)
                q = agent.W @ features
                q[ACTIONS.index("split")] = -np.inf  # not a real pair, never legal here
                learned = ACTIONS[np.argmax(q)][0].upper()
                ref = basic_strategy_action(total, usable, dealer_up)[0].upper()
                row.append(f"{learned}{'=' if learned == ref else '!'} ".rjust(4))
            lines.append("".join(row))

    lines.append("\nPAIRS (rows) vs dealer upcard 2-11 (cols)")
    lines.append(header)
    for rank in list(range(2, 11)) + [1]:
        usable = rank == 1
        total = 12 if usable else rank * 2
        row = [f"{'A,A' if usable else f'{rank},{rank}':>4} "]
        for dealer_up in range(2, 12):
            reservoir.reset(batch=1)
            x = raw_input_vec(total, usable, dealer_up).reshape(3, 1)
            state = reservoir.step(x)
            q = agent.W @ featurize(state)
            learned = ACTIONS[np.argmax(q)][0].upper()
            ref = basic_strategy_action(total, usable, dealer_up, pair_rank=rank)[0].upper()
            row.append(f"{learned}{'=' if learned == ref else '!'} ".rjust(4))
        lines.append("".join(row))

    lines.append("\n'=' agrees with basic strategy, '!' disagrees.")
    return "\n".join(lines)


def main():
    W, source = load_connectome()
    reservoir = Reservoir(W, n_inputs=3, seed=0)
    agent = QReadout(n_features=W.shape[0] + 1, lr=0.002, seed=0)
    agent.W = np.load("agent_weights.npy")
    grid = build_policy_grid(agent, reservoir)
    print(grid)
    with open("results_policy_grid.txt", "w") as f:
        f.write(grid)


if __name__ == "__main__":
    main()
