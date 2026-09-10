"""Standard casino blackjack: 6-deck shoe, dealer stands on all 17s,
blackjack pays 3:2, double down on first two cards only."""
import numpy as np

ACTIONS = ("hit", "stand", "double")
N_DECKS = 6


def make_shoe(rng):
    deck = list(range(1, 11)) + [10, 10, 10]  # A..9,10,J,Q,K -> ranks 1..10
    shoe = deck * 4 * N_DECKS
    rng.shuffle(shoe)
    return shoe


def hand_value(cards):
    total = sum(min(c, 10) for c in cards)
    aces = cards.count(1)
    usable_ace = False
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1
        usable_ace = True
    return total, usable_ace


class BlackjackEnv:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.shoe = make_shoe(self.rng)

    def _draw(self):
        if len(self.shoe) < 15:
            self.shoe = make_shoe(self.rng)
        return self.shoe.pop()

    def reset(self):
        self.player = [self._draw(), self._draw()]
        self.dealer = [self._draw(), self._draw()]
        self.done = False
        self.doubled = False
        p_bj = self._player_blackjack()
        d_bj = hand_value(self.dealer)[0] == 21
        if p_bj or d_bj:
            self.done = True
            self.natural_reward = 0.0 if (p_bj and d_bj) else (1.5 if p_bj else -1.0)
        return self._obs()

    def _obs(self):
        total, usable = hand_value(self.player)
        return total, usable, min(self.dealer[0], 10)

    def _player_blackjack(self):
        return len(self.player) == 2 and hand_value(self.player)[0] == 21

    def _dealer_play(self):
        while True:
            total, _ = hand_value(self.dealer)
            if total >= 17:
                return
            self.dealer.append(self._draw())

    def _settle(self):
        p_total, _ = hand_value(self.player)
        if p_total > 21:
            return -2.0 if self.doubled else -1.0
        self._dealer_play()
        d_total, _ = hand_value(self.dealer)
        mult = 2.0 if self.doubled else 1.0
        if d_total > 21 or p_total > d_total:
            return mult
        if p_total < d_total:
            return -mult
        return 0.0

    def step(self, action):
        assert not self.done
        reward = 0.0
        if action == "hit":
            self.player.append(self._draw())
            total, _ = hand_value(self.player)
            if total > 21 or total == 21:
                self.done = True
                reward = self._settle() if total <= 21 else -1.0
        elif action == "double":
            self.doubled = True
            self.player.append(self._draw())
            self.done = True
            total, _ = hand_value(self.player)
            reward = -2.0 if total > 21 else self._settle()
        else:  # stand
            self.done = True
            reward = self._settle()
        return self._obs(), reward, self.done
