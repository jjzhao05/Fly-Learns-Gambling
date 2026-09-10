"""Train a linear Q-learning readout on a fixed fly-connectome reservoir
to play blackjack, then compare it to published basic strategy.

Hands are simulated in batches so the (potentially whole-brain-sized)
reservoir's sparse matrix-vector product is done once per batch step
instead of once per hand."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome import load_connectome
from reservoir import Reservoir
from blackjack_env import BlackjackEnv, ACTIONS, hand_value
from agent import QReadout, featurize
from basic_strategy import basic_strategy_action

N_HANDS = 10_000
BATCH = 64
LR_REFERENCE = 0.002  # tuned for a 300-neuron reservoir
EPS_START, EPS_END, EPS_DECAY_HANDS = 1.0, 0.02, 7_000
LOG_EVERY_BATCHES = 10
EVAL_HANDS = 1_500


def scaled_lr(n_features, reference_n=301):
    """Keep the effective SGD step size roughly constant as reservoir size
    (and thus feature-vector dimension) changes, since the update step's
    norm otherwise grows with sqrt(n_features)."""
    return LR_REFERENCE * np.sqrt(reference_n / n_features)


def raw_input_vec(total, usable_ace, dealer_up):
    return np.array([total / 21.0, float(usable_ace), dealer_up / 11.0], dtype=np.float32)


def epsilon_at(hand_idx):
    frac = min(hand_idx / EPS_DECAY_HANDS, 1.0)
    return EPS_START + frac * (EPS_END - EPS_START)


def play_batch(envs, reservoir, agent, epsilon, learn=True):
    B = len(envs)
    obs = [e.reset() for e in envs]
    done = np.array([e.done for e in envs])
    reward = np.array([e.natural_reward if e.done else 0.0 for e in envs], dtype=np.float32)
    first_action = True

    reservoir.reset(batch=B)
    X = np.zeros((3, B), dtype=np.float32)
    for i in range(B):
        X[:, i] = raw_input_vec(*obs[i])
    state = reservoir.step(X)
    features = np.vstack([state, np.ones(B, dtype=np.float32)])

    while not done.all():
        active = ~done
        Q = agent.W @ features  # (3, B)
        if not first_action:
            Q[ACTIONS.index("double")] = -np.inf
        n_choices = len(ACTIONS) if first_action else 2
        greedy = np.argmax(Q, axis=0)
        random_a = np.random.randint(0, n_choices, size=B)
        explore = np.random.random(B) < epsilon
        chosen = np.where(explore, random_a, greedy)

        next_obs = list(obs)
        step_reward = np.zeros(B, dtype=np.float32)
        just_finished = np.zeros(B, dtype=bool)
        for i in range(B):
            if not active[i]:
                continue
            o, r, d = envs[i].step(ACTIONS[chosen[i]])
            next_obs[i] = o
            step_reward[i] = r
            if d:
                done[i] = True
                just_finished[i] = True

        still_active = active & ~just_finished
        X = np.zeros((3, B), dtype=np.float32)
        for i in range(B):
            if still_active[i]:
                X[:, i] = raw_input_vec(*next_obs[i])
        next_state = reservoir.step(X)
        next_features = np.vstack([next_state, np.ones(B, dtype=np.float32)])

        if learn:
            for a in range(len(ACTIONS)):
                term_mask = active & just_finished & (chosen == a)
                if term_mask.any():
                    idx = np.where(term_mask)[0]
                    td = step_reward[idx] - Q[a, idx]
                    agent.W[a] += agent.lr * (features[:, idx] @ td) / len(idx)
                cont_mask = still_active & (chosen == a)
                if cont_mask.any():
                    idx = np.where(cont_mask)[0]
                    next_q = np.max(agent.W @ next_features[:, idx], axis=0)
                    target = 0.95 * next_q
                    td = target - Q[a, idx]
                    agent.W[a] += agent.lr * (features[:, idx] @ td) / len(idx)

        reward = reward + step_reward
        features = next_features
        obs = next_obs
        first_action = False
    return reward


def evaluate_agent(agent, reservoir, n_hands, seed):
    env = BlackjackEnv(seed=seed)
    total_reward, wins, hands = 0.0, 0, 0
    for _ in range(n_hands):
        obs = env.reset()
        if env.done:
            r = env.natural_reward
            total_reward += r
            wins += r > 0
            hands += 1
            continue
        reservoir.reset(batch=1)
        first_action = True
        total, usable, dealer_up = obs
        state = reservoir.step(raw_input_vec(total, usable, dealer_up).reshape(3, 1))
        features = np.vstack([state, [[1.0]]])
        while True:
            valid = ACTIONS if first_action else ("hit", "stand")
            q = (agent.W @ features).ravel()
            idxs = [ACTIONS.index(a) for a in valid]
            action = ACTIONS[idxs[np.argmax(q[idxs])]]
            obs, reward, done = env.step(action)
            if done:
                total_reward += reward
                wins += reward > 0
                hands += 1
                break
            total, usable, dealer_up = obs
            state = reservoir.step(raw_input_vec(total, usable, dealer_up).reshape(3, 1))
            features = np.vstack([state, [[1.0]]])
            first_action = False
    return wins / hands, total_reward / hands


def evaluate_policy_fn(policy_fn, n_hands, seed):
    env = BlackjackEnv(seed=seed)
    total_reward, wins, hands = 0.0, 0, 0
    for _ in range(n_hands):
        obs = env.reset()
        if env.done:
            r = env.natural_reward
            total_reward += r
            wins += r > 0
            hands += 1
            continue
        first_action = True
        while True:
            total, usable, dealer_up = obs
            valid = ACTIONS if first_action else ("hit", "stand")
            action = policy_fn(total, usable, dealer_up, first_action)
            if action not in valid:
                action = "hit"
            obs, reward, done = env.step(action)
            if done:
                total_reward += reward
                wins += reward > 0
                hands += 1
                break
            first_action = False
    return wins / hands, total_reward / hands


def main():
    W, source = load_connectome()
    print(f"connectome source: {source}, shape={W.shape}")
    reservoir = Reservoir(W, n_inputs=3, seed=0)
    n_features = W.shape[0] + 1
    agent = QReadout(n_features=n_features, lr=scaled_lr(n_features), seed=0)
    print(f"n_features={n_features} lr={agent.lr:.2e}")

    history = []
    hands_done = 0
    n_batches = N_HANDS // BATCH
    window_reward, window_wins, window_hands = 0.0, 0, 0
    for b in range(1, n_batches + 1):
        eps = epsilon_at(hands_done)
        envs = [BlackjackEnv(seed=1000 * b + i) for i in range(BATCH)]
        rewards = play_batch(envs, reservoir, agent, eps, learn=True)
        hands_done += BATCH
        window_reward += rewards.sum()
        window_wins += int((rewards > 0).sum())
        window_hands += BATCH
        if b % LOG_EVERY_BATCHES == 0:
            win_rate = window_wins / window_hands
            ev = window_reward / window_hands
            history.append((hands_done, win_rate, ev))
            print(f"hand {hands_done:>8} eps={eps:.3f} win_rate={win_rate:.3f} ev/hand={ev:+.3f}")
            window_reward, window_wins, window_hands = 0.0, 0, 0

    agent_wr, agent_ev = evaluate_agent(agent, reservoir, EVAL_HANDS, seed=100)
    basic_wr, basic_ev = evaluate_policy_fn(basic_strategy_action, EVAL_HANDS, seed=100)

    def random_policy(total, usable, dealer_up, first_action):
        valid = ACTIONS if first_action else ("hit", "stand")
        return valid[np.random.randint(len(valid))]

    random_wr, random_ev = evaluate_policy_fn(random_policy, EVAL_HANDS, seed=100)

    print("\n=== Final evaluation (%d hands) ===" % EVAL_HANDS)
    print(f"{'Policy':<28}{'Win rate':>10}{'EV/hand':>10}")
    print(f"{'Fly-reservoir Q-agent':<28}{agent_wr:>10.3f}{agent_ev:>+10.3f}")
    print(f"{'Basic strategy':<28}{basic_wr:>10.3f}{basic_ev:>+10.3f}")
    print(f"{'Random':<28}{random_wr:>10.3f}{random_ev:>+10.3f}")

    hands, win_rates, evs = zip(*history)
    plt.figure(figsize=(8, 5))
    plt.plot(hands, win_rates, label="win rate")
    plt.axhline(basic_wr, color="gray", linestyle="--", label="basic strategy (final)")
    plt.xlabel("training hands")
    plt.ylabel("win rate (rolling window)")
    plt.title(f"Fly-connectome reservoir ({source}) + Q-learning readout")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_training_curve.png", dpi=150)

    with open("results_report.txt", "w") as f:
        f.write(f"connectome source: {source}\n")
        f.write(f"reservoir size: {W.shape[0]} neurons\n")
        f.write(f"training hands: {N_HANDS}\n\n")
        f.write(f"{'Policy':<28}{'Win rate':>10}{'EV/hand':>10}\n")
        f.write(f"{'Fly-reservoir Q-agent':<28}{agent_wr:>10.3f}{agent_ev:>+10.3f}\n")
        f.write(f"{'Basic strategy':<28}{basic_wr:>10.3f}{basic_ev:>+10.3f}\n")
        f.write(f"{'Random':<28}{random_wr:>10.3f}{random_ev:>+10.3f}\n")

    np.save("agent_weights.npy", agent.W)
    return agent, reservoir


if __name__ == "__main__":
    main()
