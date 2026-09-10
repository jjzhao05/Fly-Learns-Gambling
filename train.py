import json
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connectome import load_connectome
from reservoir import Reservoir
from blackjack_env import BlackjackEnv, ACTIONS
from agent import QReadout, featurize
from basic_strategy import basic_strategy_action

HANDS_THIS_RUN = 500_000
BATCH = 64
LR_REFERENCE = 0.002
EPS_START, EPS_END, EPS_DECAY_HANDS = 1.0, 0.02, 7_000
EPSILON_RESTART = 0.05
LOG_EVERY_BATCHES = 10
CHECKPOINT_EVERY_SECONDS = 20
EVAL_HANDS = 1_500


def checkpoint(agent, source, n_neurons, hands_done, win_rate, ev):
    np.save("agent_weights.npy.tmp.npy", agent.W)
    os.replace("agent_weights.npy.tmp.npy", "agent_weights.npy")
    with open("agent_meta.json.tmp", "w") as f:
        json.dump({"source": source, "n_neurons": int(n_neurons), "hands_done": int(hands_done),
                    "win_rate": float(win_rate), "ev": float(ev), "done": False}, f)
    os.replace("agent_meta.json.tmp", "agent_meta.json")


def scaled_lr(n_features, reference_n=301):
    return LR_REFERENCE * np.sqrt(reference_n / n_features)


def raw_input_vec(total, usable_ace, dealer_up):
    return np.array([total / 21.0, float(usable_ace), dealer_up / 11.0], dtype=np.float32)


def epsilon_at(hand_idx, start=EPS_START):
    frac = min(hand_idx / EPS_DECAY_HANDS, 1.0)
    return start + frac * (EPS_END - start)


def legal_mask(envs, active):
    mask = np.zeros((len(ACTIONS), len(envs)), dtype=bool)
    for i, e in enumerate(envs):
        if active[i]:
            for a in e.legal_actions():
                mask[ACTIONS.index(a), i] = True
    return mask


def play_batch(envs, reservoir, agent, epsilon, learn=True):
    B = len(envs)
    obs = [e.reset() for e in envs]
    done = np.array([e.done for e in envs])
    reward = np.array([e.natural_reward if e.done else 0.0 for e in envs], dtype=np.float32)

    reservoir.reset(batch=B)
    X = np.zeros((3, B), dtype=np.float32)
    for i in range(B):
        X[:, i] = raw_input_vec(*obs[i])
    state = reservoir.step(X)
    features = np.vstack([state, np.ones(B, dtype=np.float32)])

    while not done.all():
        active = ~done
        mask = legal_mask(envs, active)
        Q = agent.W @ features
        Qm = np.where(mask, Q, -np.inf)
        greedy = np.argmax(Qm, axis=0)
        random_a = np.array([np.random.choice(np.where(mask[:, i])[0]) if active[i] else 0
                              for i in range(B)])
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
        next_mask = legal_mask(envs, still_active)

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
                    next_q_all = np.where(next_mask[:, idx], agent.W @ next_features[:, idx], -np.inf)
                    next_q = np.max(next_q_all, axis=0)
                    target = step_reward[idx] + 0.95 * next_q
                    td = target - Q[a, idx]
                    agent.W[a] += agent.lr * (features[:, idx] @ td) / len(idx)

        reward = reward + step_reward
        features = next_features
        obs = next_obs
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
        total, usable, dealer_up = obs
        state = reservoir.step(raw_input_vec(total, usable, dealer_up).reshape(3, 1))
        features = np.vstack([state, [[1.0]]])
        hand_reward = 0.0
        while True:
            valid = env.legal_actions()
            q = (agent.W @ features).ravel()
            idxs = [ACTIONS.index(a) for a in valid]
            action = ACTIONS[idxs[np.argmax(q[idxs])]]
            obs, reward, done = env.step(action)
            hand_reward += reward
            if done:
                total_reward += hand_reward
                wins += hand_reward > 0
                hands += 1
                break
            total, usable, dealer_up = obs
            state = reservoir.step(raw_input_vec(total, usable, dealer_up).reshape(3, 1))
            features = np.vstack([state, [[1.0]]])
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
        hand_reward = 0.0
        while True:
            total, usable, dealer_up = obs
            valid = env.legal_actions()
            first_action = len(env.player) == 2
            pair_rank = env.player[0] if (first_action and env.player[0] == env.player[1]) else None
            action = policy_fn(total, usable, dealer_up, first_action, pair_rank)
            if action not in valid:
                action = "hit"
            obs, reward, done = env.step(action)
            hand_reward += reward
            if done:
                total_reward += hand_reward
                wins += hand_reward > 0
                hands += 1
                break
    return wins / hands, total_reward / hands


def main():
    W, source = load_connectome()
    print(f"connectome source: {source}, shape={W.shape}")
    reservoir = Reservoir(W, n_inputs=3, seed=0)
    n_features = W.shape[0] + 1
    agent = QReadout(n_features=n_features, lr=scaled_lr(n_features), seed=0)

    hands_done = 0
    if os.path.exists("agent_weights.npy") and os.path.exists("agent_meta.json"):
        with open("agent_meta.json") as f:
            prev_meta = json.load(f)
        if prev_meta.get("source") == source and prev_meta.get("n_neurons") == W.shape[0]:
            old_W = np.load("agent_weights.npy")
            agent.W[:old_W.shape[0]] = old_W
            hands_done = int(prev_meta.get("hands_done", 0))
            print(f"resuming from checkpoint at hand {hands_done} "
                  f"(actions {old_W.shape[0]} -> {agent.W.shape[0]})")

    print(f"n_features={n_features} lr={agent.lr:.2e}")

    resume_start = hands_done
    history = []
    n_batches = (hands_done + HANDS_THIS_RUN) // BATCH
    start_batch = hands_done // BATCH
    window_reward, window_wins, window_hands = 0.0, 0, 0
    last_checkpoint = 0.0
    for b in range(start_batch + 1, n_batches + 1):
        if EPSILON_RESTART is not None:
            eps = epsilon_at(hands_done - resume_start, start=EPSILON_RESTART)
        else:
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
            if time.time() - last_checkpoint > CHECKPOINT_EVERY_SECONDS:
                checkpoint(agent, source, W.shape[0], hands_done, win_rate, ev)
                last_checkpoint = time.time()

    agent_wr, agent_ev = evaluate_agent(agent, reservoir, EVAL_HANDS, seed=100)
    basic_wr, basic_ev = evaluate_policy_fn(basic_strategy_action, EVAL_HANDS, seed=100)

    def random_policy(total, usable, dealer_up, first_action, pair_rank=None):
        valid = ACTIONS if first_action else ("hit", "stand")
        return valid[np.random.randint(len(valid))]

    random_wr, random_ev = evaluate_policy_fn(random_policy, EVAL_HANDS, seed=100)

    print("\n=== Final evaluation (%d hands) ===" % EVAL_HANDS)
    print(f"{'Policy':<28}{'Win rate':>10}{'EV/hand':>10}")
    print(f"{'Fly-reservoir Q-agent':<28}{agent_wr:>10.3f}{agent_ev:>+10.3f}")
    print(f"{'Basic strategy':<28}{basic_wr:>10.3f}{basic_ev:>+10.3f}")
    print(f"{'Random':<28}{random_wr:>10.3f}{random_ev:>+10.3f}")

    hands, win_rates, _ = zip(*history)
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
        f.write(f"training hands: {hands_done}\n\n")
        f.write(f"{'Policy':<28}{'Win rate':>10}{'EV/hand':>10}\n")
        f.write(f"{'Fly-reservoir Q-agent':<28}{agent_wr:>10.3f}{agent_ev:>+10.3f}\n")
        f.write(f"{'Basic strategy':<28}{basic_wr:>10.3f}{basic_ev:>+10.3f}\n")
        f.write(f"{'Random':<28}{random_wr:>10.3f}{random_ev:>+10.3f}\n")

    checkpoint(agent, source, W.shape[0], hands_done, agent_wr, agent_ev)
    with open("agent_meta.json") as f:
        meta = json.load(f)
    meta["done"] = True
    with open("agent_meta.json.tmp", "w") as f:
        json.dump(meta, f)
    os.replace("agent_meta.json.tmp", "agent_meta.json")
    return agent, reservoir


if __name__ == "__main__":
    main()
