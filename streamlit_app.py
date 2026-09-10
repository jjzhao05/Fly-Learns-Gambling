import base64
import json
import os
import time
import numpy as np
import streamlit as st

from basic_strategy import basic_strategy_action
from connectome import load_connectome
from reservoir import Reservoir
from agent import QReadout, featurize
from blackjack_env import BlackjackEnv, ACTIONS
from train import raw_input_vec
import brain_data

st.set_page_config(page_title="Fly Learns Blackjack", layout="wide")
RANK = lambda v: "A" if v == 1 else str(v)
CAT_COLOR = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a", 3: "#8a8578"}

TABLE_CSS = """
<style>
.felt{background:radial-gradient(120% 140% at 50% -10%, #2a6349, #1f4d3a);
  border:1px solid #173d2d;border-radius:12px;padding:20px 16px;color:#eef7f0;
  display:flex;flex-direction:column;justify-content:flex-start;gap:14px;height:370px;
  overflow:hidden}
.row{display:flex;flex-direction:column;gap:6px;align-items:center;max-width:100%}
.label{font-size:.7rem;letter-spacing:.08em;text-transform:uppercase;color:#bfe3cf;opacity:.85}
.cards{display:flex;gap:6px;height:60px;align-items:center;justify-content:center;
  flex-wrap:nowrap;overflow:hidden;perspective:400px;max-width:100%}
.card{width:40px;min-width:20px;flex:0 1 40px;height:58px;border-radius:6px;background:#fbf9f3;
  color:#1c1f1c;display:flex;align-items:center;justify-content:center;font-weight:700;
  font-size:1.1rem;box-shadow:0 2px 5px rgba(0,0,0,.35)}
.card.back{background:repeating-linear-gradient(45deg,#8a3b3b,#8a3b3b 4px,#7a3232 4px,#7a3232 8px);color:transparent}
.card.deal{animation:dealIn .35s ease-out backwards}
.card.flip{animation:flipReveal .4s ease-in-out backwards}
@keyframes dealIn{0%{transform:translate(-20px,-14px) rotate(-10deg);opacity:0}100%{transform:none;opacity:1}}
@keyframes flipReveal{0%{transform:scaleX(0)}45%{transform:scaleX(0)}100%{transform:scaleX(1)}}
.total{font-size:.82rem;background:rgba(0,0,0,.28);border-radius:999px;padding:3px 12px;align-self:center;min-height:1.1em}
.action{min-height:1.9rem;font-weight:700;letter-spacing:.03em;text-transform:uppercase;font-size:.82rem;color:#ffd98a;text-align:center}
.compare{min-height:1rem;font-size:.7rem;text-align:center;opacity:.9}
.compare.match{color:#8fe3a8} .compare.diff{color:#f2b880}
.outcome{text-align:center;font-weight:700;font-size:.85rem;padding:4px 10px;border-radius:8px;display:inline-block}
.outcome.win{background:#3f8f5c;color:#08210f} .outcome.lose{background:#c1503f;color:#2a0906} .outcome.push{background:#3a4a42;color:#eef1ec}
</style>
"""


def _cards_html(cards, new_count, flip_index=None):
    n = len(cards)
    html, dealt = "", 0
    for i, c in enumerate(cards):
        if flip_index is not None and i == flip_index:
            html += f'<div class="card flip" style="animation-delay:0s">{RANK(c)}</div>'
        elif i >= n - new_count:
            html += f'<div class="card deal" style="animation-delay:{dealt * 0.12:.2f}s">{RANK(c)}</div>'
            dealt += 1
        else:
            html += f'<div class="card">{RANK(c)}</div>'
    return html


def table_html(dealer_cards, dealer_back, player_cards, total_text="", dealer_total_text="",
                action_text="", outcome=None, new_dealer=0, new_player=0, flip_hole=False,
                compare_text="", compare_match=None):
    d_html = _cards_html(dealer_cards, new_dealer, flip_index=1 if flip_hole else None)
    if dealer_back:
        d_html += '<div class="card back deal">?</div>'
    p_html = _cards_html(player_cards, new_player)
    action_html = f'<div class="outcome {outcome[0]}">{outcome[1]}</div>' if outcome else action_text
    compare_html = (f'<div class="compare {"match" if compare_match else "diff"}">'
                     f'{compare_text}</div>')
    return (TABLE_CSS +
            f'<div class="felt"><div class="row"><div class="label">Dealer</div>'
            f'<div class="cards">{d_html}</div>'
            f'<div class="total">{dealer_total_text}</div></div>'
            f'<div class="action">{action_html}</div>{compare_html}'
            f'<div class="row"><div class="label">Fly</div>'
            f'<div class="cards">{p_html}</div>'
            f'<div class="total">{total_text}</div></div></div>')


LOG_CSS = """
<style>
.logwrap{max-height:220px;overflow-y:auto;border:1px solid #2b3b33;border-radius:8px;
  background:rgba(0,0,0,.15)}
.logempty{padding:10px 12px;font-size:.75rem;opacity:.55}
.logrow{display:grid;grid-template-columns:2.6em 1fr 8em 8em 6.5em;gap:10px;padding:5px 12px;font-size:.75rem;
  border-bottom:1px solid rgba(255,255,255,.06);align-items:baseline}
.logrow:last-child{border-bottom:none}
.logrow .hand{opacity:.5}
.logrow .hand-state{color:#dfe9e2}
.logrow .fly{}
.logrow .basic{opacity:.75}
.logrow .result{opacity:.55;font-size:.7rem}
.logrow.match .fly{color:#8fe3a8}
.logrow.diff .fly{color:#f2b880}
.logrow .result.win{color:#8fe3a8;opacity:1}
.logrow .result.lose{color:#f28f80;opacity:1}
.logrow .result.push{opacity:.75}
</style>
"""


def render_log_header(entries, session_value, hands_played, basic_ev):
    avg = session_value / hands_played if hands_played else 0.0
    value_part = (f" &nbsp;&nbsp; **Net: {session_value:+.1f}** "
                  f"({avg:+.3f}/hand, basic strategy ≈ {basic_ev:+.3f}/hand)")
    if not entries:
        return "**Decision log** &nbsp;·&nbsp; fly vs. basic strategy" + value_part
    matches = sum(e["match"] for e in entries)
    pct = matches / len(entries)
    return (f"**Decision log** &nbsp;·&nbsp; fly vs. basic strategy &nbsp;&nbsp; "
            f"**Running accuracy: {pct:.0%}** ({matches}/{len(entries)})" + value_part)


def render_log_html(entries):
    if not entries:
        return LOG_CSS + '<div class="logwrap"><div class="logempty">No decisions yet -- deal a hand.</div></div>'
    rows = "".join(
        f'<div class="logrow {"match" if e["match"] else "diff"}">'
        f'<span class="hand">#{e["hand"]}</span>'
        f'<span class="hand-state">{e["total"]}{" soft" if e["usable"] else ""} vs {RANK(e["dealer_up"])}</span>'
        f'<span class="fly">fly: {e["action"].upper()}</span>'
        f'<span class="basic">basic: {e["basic"].upper()}</span>'
        f'<span class="result {e["result"][0] if e["result"] else ""}">'
        f'{e["result"][1] if e["result"] else "…"}</span></div>'
        for e in reversed(entries)
    )
    return LOG_CSS + f'<div class="logwrap">{rows}</div>'


BASIC_EV = -0.005


@st.cache_resource(show_spinner="Loading connectome...")
def load_reservoir(n_neurons_hint):
    W, source = load_connectome()
    reservoir = Reservoir(W, n_inputs=3, seed=0)
    pos_x, pos_y, cat = brain_data.full_brain_positions(W.shape[0])
    return reservoir, source, pos_x, pos_y, cat


def load_weights():
    if not os.path.exists("agent_weights.npy"):
        return None, {}
    for attempt in range(5):
        try:
            W = np.load("agent_weights.npy")
            meta = json.load(open("agent_meta.json")) if os.path.exists("agent_meta.json") else {}
            return W, meta
        except (ValueError, EOFError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(0.2)


TOPK_ACTIVE = 1500
SUBSTEPS = 48
STEP_HOLD = 0.35
COSMETIC_LEAK = 0.25
SPEED_OPTIONS = {"Very slow": 0.2, "Slow": 0.35, "Normal": 0.6, "Fast": 1.0, "Very fast": 1.6}
IFRAME_WARMUP = 0.45


def brain_frame(state, baseline=None):
    ref = state if baseline is None else (state - baseline)
    thresh, scale = (0.05, 0.4) if baseline is None else (0.02, 0.15)
    top_idx = np.argsort(-np.abs(ref))[:TOPK_ACTIVE]
    vals = ref[top_idx]
    keep = np.abs(vals) > thresh
    idx = top_idx[keep]
    mag = np.minimum(1.0, np.abs(vals[keep]) / scale)
    size = 2.2 + mag * 7
    return {"idx": [int(i) for i in idx], "size": [round(float(s), 2) for s in size]}


def render_brain_html(pos_x_b64, pos_y_b64, cat_b64, palette, frames, frame_delay_ms):
    return f"""
<style>html,body{{background:#0d1310;margin:0}}</style>
<div id="brainplot" style="width:100%;height:460px;background:#0d1310;"></div>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<script>
function b64f32(b64) {{
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}}
function b64u8(b64) {{
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}}
const posX = b64f32("{pos_x_b64}");
const posY = b64f32("{pos_y_b64}");
const cats = b64u8("{cat_b64}");
const palette = {json.dumps(palette)};

const bgStep = Math.max(1, Math.floor(cats.length / 40000));
const bgX = [], bgY = [], bgColors = [];
for (let i = 0; i < cats.length; i += bgStep) {{
  bgX.push(posX[i]); bgY.push(posY[i]); bgColors.push(palette[cats[i]]);
}}

const base = {{x: bgX, y: bgY, mode: "markers", type: "scattergl",
              marker: {{size: 2.0, color: bgColors, opacity: 0.75, line: {{width: 0}}}}}};
const active = {{x: [], y: [], mode: "markers", type: "scattergl",
                marker: {{size: [], color: "#ffcf6b", opacity: 0.95, line: {{width: 0}}}}}};
Plotly.newPlot("brainplot", [base, active], {{
  paper_bgcolor: "#0d1310", plot_bgcolor: "#0d1310",
  xaxis: {{visible: false}}, yaxis: {{visible: false, scaleanchor: "x"}},
  margin: {{l: 0, r: 0, t: 0, b: 0}}, showlegend: false
}}, {{displayModeBar: false, responsive: true}});

const frames = {json.dumps(frames)};
function applyFrame(f) {{
  const xs = f.idx.map(j => posX[j]);
  const ys = f.idx.map(j => posY[j]);
  Plotly.restyle("brainplot", {{x: [xs], y: [ys], "marker.size": [f.size]}}, [1]);
}}
let i = 0;
if (frames.length > 0) applyFrame(frames[0]);
function stepFrame() {{
  i++;
  if (i >= frames.length) return;
  applyFrame(frames[i]);
  setTimeout(stepFrame, {frame_delay_ms});
}}
if (frames.length > 1) setTimeout(stepFrame, {frame_delay_ms});
</script>
"""


st.title("Fly Learns Blackjack")
st.caption("House rules: 6-deck shoe, dealer stands on 17 and hits below it, blackjack "
           "pays 3:2, and doubling down is only allowed on the first two cards.")

W_arr, meta = load_weights()
if W_arr is None:
    st.warning("No agent_weights.npy yet. Run `python train.py` first.")
    st.stop()

n_neurons = meta.get("n_neurons", W_arr.shape[1] - 1)
reservoir, source, pos_x, pos_y, cat = load_reservoir(n_neurons)
PALETTE = [CAT_COLOR[i] for i in range(4)]
pos_x_b64 = base64.b64encode(pos_x.astype(np.float32).tobytes()).decode("ascii")
pos_y_b64 = base64.b64encode(pos_y.astype(np.float32).tobytes()).decode("ascii")
cat_b64 = base64.b64encode(np.asarray(cat, dtype=np.uint8).tobytes()).decode("ascii")
IDLE_FRAME = brain_frame(np.zeros(n_neurons, dtype=np.float32))

if "hand_seed" not in st.session_state:
    st.session_state.hand_seed = 0
if "decision_log" not in st.session_state:
    st.session_state.decision_log = []
if "continuous_play" not in st.session_state:
    st.session_state.continuous_play = False
if "session_value" not in st.session_state:
    st.session_state.session_value = 0.0

col1, col3 = st.columns(2)
col1.metric("Hands trained", f"{meta.get('hands_done', 0):,}")
hands_slot = col3.empty()
hands_slot.metric("Hands played", f"{st.session_state.hand_seed:,}")

legend = " &nbsp;&nbsp; ".join(f"<span style='color:{c}'>●</span> {name}"
                                for name, c in zip(brain_data.CATS, CAT_COLOR.values()))
st.markdown(legend + " &nbsp;&nbsp; <span style='color:#ffcf6b'>●</span> firing now",
            unsafe_allow_html=True)

table_col, brain_col = st.columns([1, 1.3])
table_slot = table_col.empty()
brain_slot = brain_col.empty()
table_slot.markdown(table_html([], False, []), unsafe_allow_html=True)
speed_label = table_col.select_slider("Blackjack speed", options=list(SPEED_OPTIONS), value="Normal")
speed = SPEED_OPTIONS[speed_label]
with brain_slot.container():
    st.iframe(render_brain_html(pos_x_b64, pos_y_b64, cat_b64, PALETTE, [IDLE_FRAME], 1000),
              height=470)

btn_deal, btn_cont, _ = st.columns([1, 1, 3])
deal_clicked = btn_deal.button("Deal a hand", type="primary")
cont_label = "⏹ Stop continuous play" if st.session_state.continuous_play else "▶ Continuous play"
if btn_cont.button(cont_label):
    st.session_state.continuous_play = not st.session_state.continuous_play

header_slot = st.empty()
header_slot.markdown(render_log_header(st.session_state.decision_log, st.session_state.session_value,
                                        st.session_state.hand_seed, BASIC_EV),
                      unsafe_allow_html=True)
log_slot = st.empty()
log_slot.markdown(render_log_html(st.session_state.decision_log), unsafe_allow_html=True)


def hand_total(cards):
    total, aces = sum(min(c, 10) for c in cards), cards.count(1)
    while aces > 0 and total + 10 <= 21:
        total += 10
        aces -= 1
    return total


def outcome_badge(reward, natural):
    if reward > 0:
        return ("win", "BLACKJACK +1.5" if natural else f"WIN {reward:+.1f}")
    if reward < 0:
        return ("lose", f"LOSS {reward:+.1f}")
    return ("push", "PUSH")


if deal_clicked or st.session_state.continuous_play:
    agent = QReadout(n_features=n_neurons + 1, lr=1.0, seed=0)
    agent.W[:W_arr.shape[0]] = W_arr
    env = BlackjackEnv(seed=st.session_state.hand_seed)
    st.session_state.hand_seed += 1
    hand_no = st.session_state.hand_seed
    hands_slot.metric("Hands played", f"{hand_no:,}")
    obs = env.reset()
    player_cards = list(env.player)
    dealer_up = env.dealer[0]

    deal_pause = 0.35 / speed
    deal_steps = [
        ([], player_cards[:1], False, 0, 1),
        ([dealer_up], player_cards[:1], False, 1, 0),
        ([dealer_up], player_cards, False, 0, 1),
        ([dealer_up], player_cards, True, 0, 0),
    ]
    for d_cards, p_cards, d_back, nd, np_ in deal_steps:
        tot = f"Total {hand_total(p_cards)}" if len(p_cards) == 2 else ""
        table_slot.markdown(table_html(d_cards, d_back, p_cards, tot,
                                        new_dealer=nd, new_player=np_),
                             unsafe_allow_html=True)
        time.sleep(deal_pause)

    if env.done:
        with brain_slot.container():
            st.iframe(render_brain_html(pos_x_b64, pos_y_b64, cat_b64, PALETTE,
                                         [IDLE_FRAME], 1000),
                      height=470)
        time.sleep(IFRAME_WARMUP + 0.4 / speed)
        table_slot.markdown(table_html(env.dealer, False, player_cards,
                                        f"Total {hand_total(player_cards)}",
                                        f"Total {hand_total(env.dealer)}", "",
                                        outcome_badge(env.natural_reward, True),
                                        flip_hole=True),
                             unsafe_allow_html=True)
        st.session_state.session_value += env.natural_reward
        header_slot.markdown(render_log_header(st.session_state.decision_log, st.session_state.session_value,
                                                 st.session_state.hand_seed, BASIC_EV),
                              unsafe_allow_html=True)
    else:
        time.sleep(0.4 / speed)

        reservoir.reset(batch=1)
        total, usable, dealer_up = obs
        brain_frames = []
        events = []
        pending_entries = []

        frame_delay_ms = max(8, int(1000 * STEP_HOLD / speed / SUBSTEPS))
        decision_wait = frame_delay_ms * SUBSTEPS / 1000.0

        def pad_frames(seconds):
            n = max(0, round(seconds * 1000 / frame_delay_ms))
            brain_frames.extend([brain_frames[-1] if brain_frames else IDLE_FRAME] * n)

        def append_settle_events(cards, dealer_hand, hand_reward):
            events.append((0.4 / speed,
                            table_html(dealer_hand[:2], False, cards,
                                       f"Total {hand_total(cards)}",
                                       f"Total {hand_total(dealer_hand[:2])}",
                                       "dealer flips…", flip_hole=True),
                            None, None))
            pad_frames(0.4 / speed)
            shown = 2
            while shown < len(dealer_hand):
                shown += 1
                events.append((0.5 / speed,
                                table_html(dealer_hand[:shown], False, cards,
                                           f"Total {hand_total(cards)}",
                                           f"Total {hand_total(dealer_hand[:shown])}",
                                           "dealer hits", new_dealer=1),
                                None, None))
                pad_frames(0.5 / speed)
            events.append((0.4 / speed,
                            table_html(dealer_hand, False, cards,
                                       f"Total {hand_total(cards)}",
                                       f"Total {hand_total(dealer_hand)}", "",
                                       outcome_badge(hand_reward, False)),
                            None, hand_reward))
            pad_frames(0.4 / speed)

        while True:
            x = raw_input_vec(total, usable, dealer_up).reshape(3, 1)
            prev_state = reservoir.state.ravel().copy()

            sim_state = prev_state.reshape(-1, 1).copy()
            micro_leak = COSMETIC_LEAK
            for _ in range(SUBSTEPS - 1):
                pre = reservoir.W @ sim_state + reservoir.W_in @ x
                sim_state = (1 - micro_leak) * sim_state + micro_leak * np.tanh(pre)
                brain_frames.append(brain_frame(sim_state.ravel(), baseline=prev_state))

            state = reservoir.step(x).ravel()
            brain_frames.append(brain_frame(state, baseline=prev_state))

            features = featurize(state)
            valid = env.legal_actions()
            first = len(env.player) == 2
            pair_rank = env.player[0] if (first and env.player[0] == env.player[1]) else None
            q = agent.W @ features
            idxs = [ACTIONS.index(a) for a in valid]
            action = ACTIONS[idxs[int(np.argmax(q[idxs]))]]

            basic_action = basic_strategy_action(total, usable, dealer_up, first_action=first,
                                                  pair_rank=pair_rank)
            log_entry = {"hand": hand_no, "total": total, "usable": usable, "dealer_up": dealer_up,
                         "action": action, "basic": basic_action, "match": action == basic_action,
                         "result": None}
            pending_entries.append(log_entry)
            events.append((decision_wait,
                            table_html([dealer_up], True, player_cards,
                                       f"Total {hand_total(player_cards)}", "",
                                       f"deciding… → {action.upper()}",
                                       compare_text=f"basic strategy says {basic_action.upper()}",
                                       compare_match=(action == basic_action)),
                            log_entry, None))

            obs, reward, done = env.step(action)

            if action == "split":
                player_cards = list(env.player)
                events.append((0.5 / speed,
                                table_html([dealer_up], True, player_cards,
                                           f"Total {hand_total(player_cards)}", "",
                                           "SPLIT!", new_player=2),
                                None, None))
                pad_frames(0.5 / speed)
                if env.aces_split:
                    (cards_a, dealer_a, reward_a), (cards_b, dealer_b, reward_b) = env.split_ace_detail
                    append_settle_events(cards_a, dealer_a, reward_a)
                    events.append((0.5 / speed,
                                    table_html([dealer_b[0]], True, cards_b,
                                               f"Total {hand_total(cards_b)}", "",
                                               "Hand 2", new_player=2),
                                    None, None))
                    pad_frames(0.5 / speed)
                    append_settle_events(cards_b, dealer_b, reward_b)
                    for e in pending_entries:
                        e["result"] = outcome_badge(reward, False)
                    pending_entries = []
                    break
            elif action in ("hit", "double"):
                player_cards = list(env.player)
                events.append((0.5 / speed,
                                table_html([dealer_up], True, player_cards,
                                           f"Total {hand_total(player_cards)}", "",
                                           action.upper(), new_player=1,
                                           compare_text=f"basic strategy said {basic_action.upper()}",
                                           compare_match=(action == basic_action)),
                                None, None))
                pad_frames(0.5 / speed)

            if env.hand_ended:
                append_settle_events(player_cards, env.dealer, reward)
                for e in pending_entries:
                    e["result"] = outcome_badge(reward, False)
                pending_entries = []
                if done:
                    break
                player_cards = list(env.player)
                total, usable, dealer_up = obs
                events.append((0.5 / speed,
                                table_html([dealer_up], True, player_cards,
                                           f"Total {hand_total(player_cards)}", "",
                                           "Hand 2", new_player=2),
                                None, None))
                pad_frames(0.5 / speed)
                continue

            total, usable, dealer_up = obs

        with brain_slot.container():
            st.iframe(render_brain_html(pos_x_b64, pos_y_b64, cat_b64, PALETTE,
                                         brain_frames, frame_delay_ms),
                      height=470)
        time.sleep(IFRAME_WARMUP)
        for wait_seconds, html, log_entry, reward_delta in events:
            time.sleep(wait_seconds)
            table_slot.markdown(html, unsafe_allow_html=True)
            if log_entry is not None:
                st.session_state.decision_log.append(log_entry)
                st.session_state.decision_log = st.session_state.decision_log[-50:]
                log_slot.markdown(render_log_html(st.session_state.decision_log), unsafe_allow_html=True)
            if reward_delta is not None:
                st.session_state.session_value += reward_delta
                header_slot.markdown(render_log_header(st.session_state.decision_log,
                                                         st.session_state.session_value,
                                                         st.session_state.hand_seed, BASIC_EV),
                                      unsafe_allow_html=True)

    if st.session_state.continuous_play:
        time.sleep(0.5 / speed)
        st.rerun()
