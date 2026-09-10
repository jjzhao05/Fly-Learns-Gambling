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


def basic_strategy_action(total, usable_ace, dealer_up, first_action=True):
    table = SOFT if usable_ace and total >= 13 else HARD
    action = table.get(total, {}).get(dealer_up, "stand" if total >= 17 else "hit")
    if action == "double" and not first_action:
        action = "hit"
    return action
