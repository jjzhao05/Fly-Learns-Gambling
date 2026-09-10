import numpy as np

ACTIONS = ("hit", "stand", "double", "split")
N_DECKS = 6


def make_shoe(rng):
    deck = list(range(1, 11)) + [10, 10, 10]
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
        self.pending_hand = None
        self.split_used = False
        self.aces_split = False
        self.hand_ended = False
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

    def legal_actions(self):
        acts = ["hit", "stand"]
        if len(self.player) == 2:
            acts.append("double")
            if not self.split_used and self.player[0] == self.player[1]:
                acts.append("split")
        return tuple(acts)

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

    def _advance(self, carry=0.0):
        self.hand_ended = True
        reward = carry + self._settle()
        if self.pending_hand is None:
            self.done = True
            return self._obs(), reward, True
        self.player = self.pending_hand
        self.pending_hand = None
        self.doubled = False
        self.dealer = [self.dealer[0], self._draw()]
        if self.aces_split:
            return self._advance(carry=reward)
        return self._obs(), reward, False

    def step(self, action):
        assert not self.done
        self.hand_ended = False
        if action == "split":
            self.split_used = True
            c1, c2 = self.player
            self.pending_hand = [c2, self._draw()]
            self.player = [c1, self._draw()]
            if c1 == 1:
                self.aces_split = True
                return self._advance()
            return self._obs(), 0.0, False
        if action == "hit":
            self.player.append(self._draw())
            total, _ = hand_value(self.player)
            if total >= 21:
                return self._advance()
            return self._obs(), 0.0, False
        if action == "double":
            self.doubled = True
            self.player.append(self._draw())
            return self._advance()
        return self._advance()
