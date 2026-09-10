"""Published basic strategy for a 6-deck, dealer-stands-on-17 game
(hit/stand/double only, no splits since the agent can't split)."""

# hard totals: {player_total: {dealer_upcard(2-11): action}}
HARD = {
    8: {u: "hit" for u in range(2, 12)},
    9: {u: ("double" if 3 <= u <= 6 else "hit") for u in range(2, 12)},
    10: {u: ("double" if u <= 9 else "hit") for u in range(2, 12)},
    11: {u: "double" for u in range(2, 12)},
    12: {u: ("stand" if 4 <= u <= 6 else "hit") for u in range(2, 12)},
    13: {u: ("stand" if u <= 6 else "hit") for u in range(2, 12)},
    14: {u: ("stand" if u <= 6 else "hit") for u in range(2, 12)},
    15: {u: ("stand" if u <= 6 else "hit") for u in range(2, 12)},
    16: {u: ("stand" if u <= 6 else "hit") for u in range(2, 12)},
    17: {u: "stand" for u in range(2, 12)},
}
for t in range(2, 8):
    HARD[t] = {u: "hit" for u in range(2, 12)}
for t in range(18, 22):
    HARD[t] = {u: "stand" for u in range(2, 12)}

# soft totals (usable ace): {player_total: {dealer_upcard: action}}
SOFT = {
    13: {u: ("double" if u in (5, 6) else "hit") for u in range(2, 12)},
    14: {u: ("double" if u in (5, 6) else "hit") for u in range(2, 12)},
    15: {u: ("double" if 4 <= u <= 6 else "hit") for u in range(2, 12)},
    16: {u: ("double" if 4 <= u <= 6 else "hit") for u in range(2, 12)},
    17: {u: ("double" if 3 <= u <= 6 else "hit") for u in range(2, 12)},
    18: {u: ("double" if 3 <= u <= 6 else ("stand" if u <= 8 else "hit")) for u in range(2, 12)},
    19: {u: "stand" for u in range(2, 12)},
    20: {u: "stand" for u in range(2, 12)},
    21: {u: "stand" for u in range(2, 12)},
}
for t in range(12, 13):
    SOFT[t] = {u: "hit" for u in range(2, 12)}

# pairs (double after split assumed): {rank: {dealer_upcard: action}}, rank 1 = A,A
PAIRS = {
    1: {u: "split" for u in range(2, 12)},
    10: {u: "stand" for u in range(2, 12)},
    9: {u: ("stand" if u in (7, 10, 11) else "split") for u in range(2, 12)},
    8: {u: "split" for u in range(2, 12)},
    7: {u: ("split" if u <= 7 else "hit") for u in range(2, 12)},
    6: {u: ("split" if u <= 6 else "hit") for u in range(2, 12)},
    5: {u: ("double" if u <= 9 else "hit") for u in range(2, 12)},
    4: {u: ("split" if u in (5, 6) else "hit") for u in range(2, 12)},
    3: {u: ("split" if u <= 7 else "hit") for u in range(2, 12)},
    2: {u: ("split" if u <= 7 else "hit") for u in range(2, 12)},
}


def basic_strategy_action(total, usable_ace, dealer_up, first_action=True, pair_rank=None):
    if first_action and pair_rank is not None:
        action = PAIRS.get(pair_rank, {}).get(dealer_up)
        if action:
            return action
    table = SOFT if usable_ace and total >= 13 else HARD
    action = table.get(total, {}).get(dealer_up, "stand" if total >= 17 else "hit")
    if action == "double" and not first_action:
        action = "hit"
    return action
