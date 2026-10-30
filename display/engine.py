"""Play one MARSHAL game move by move, the way ROLL trains on it.

The games themselves are MARSHAL's own classes (`roll/agentic/env/*`), so the
rules, observations, legal actions, built-in opponents and rewards are exactly
the training ones. What decides what a model *sees* and how its answer is
*judged* lives in `roll/agentic/rollout/env_manager.py`, which needs torch and
Ray. This module ports those parts without them:

- the conversation each player gets (`EnvManager._format_messages`), rendered
  with the chat template EnvManager installs, which keeps earlier <think> blocks;
- the strict answer check (`_parse_response`, `_extract_map_valid_actions`) and
  the forfeit rules (`get_env_input`): a bad format, a response of 4096 tokens or
  more, or fewer than 1000 tokens left in the sequence loses the game;
- the per-turn rewards MARSHAL's turn-level advantage estimator is trained on
  (`_log_env_state`, `compute_length_penalty`, `format_penalty`).

A finished game is a plain dict (see `Game.record`), saved as
`*.marshal-game.json` and replayed by the display.
"""

import json
import re
import threading
import time
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------- training settings
# From AgenticConfig defaults and examples/*/agentic_val_*_selfplay.yaml.
FORMAT_PENALTY = 0.05  # train_env_manager.format_penalty
ACTION_SEP = "||"
SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]
LENGTH_REWARD = {"upper": 0.0, "lower": 0.5, "min_len": 11, "max_len": 2048, "coef": 1}
OVERLONG_RESPONSE = 4096  # get_env_input: token_length >= 4096 forfeits
MIN_TOKENS_LEFT = 1000  # get_env_input: token_left <= 1000 forfeits
MAX_ACTIONS_PER_TRAJ = 50
DEFAULT_SEQUENCE_LENGTH = 32768  # paper config (the bring-up runs used 4096)
DEFAULT_MAX_NEW_TOKENS = 4096  # paper config (the bring-up runs used 1024)

GAMES = {
    "tictactoe": {
        "title": "Tic-Tac-Toe", "family": "Competitive · perfect information", "split": "training",
        "bot": "mcts", "bot_label": "MCTS bot", "marks": ["X", "O"],
        "blurb": "Three in a row on a 3×3 grid. One of MARSHAL's three self-play training games.",
    },
    "kuhn_poker": {
        "title": "Kuhn Poker", "family": "Competitive · imperfect information", "split": "training",
        "bot": "cfr", "bot_label": "CFR equilibrium bot", "marks": ["P0", "P1"],
        "blurb": "Three cards (J < Q < K), one each, one round of betting. A MARSHAL training game: "
                 "bluffing and reading the other player's bets.",
    },
    "hanabi": {
        "title": "Mini Hanabi", "family": "Cooperative · imperfect information", "split": "training",
        "bot": None, "bot_label": None, "marks": ["P0", "P1"],
        "blurb": "Both players win or lose together. You see your partner's cards but not your own, "
                 "and spend hints to build the fireworks. A MARSHAL training game.",
    },
    "connect_four": {
        "title": "Connect Four", "family": "Competitive · perfect information", "split": "held-out",
        "bot": "mcts", "bot_label": "MCTS bot", "marks": ["X", "O"],
        "blurb": "Four in a row on a 6×7 grid. Held out: MARSHAL never trains on it.",
    },
    "leduc_poker": {
        "title": "Leduc Hold'em", "family": "Competitive · imperfect information", "split": "held-out",
        "bot": "ne", "bot_label": "Nash-equilibrium bot", "marks": ["P0", "P1"],
        "blurb": "Six cards, a private card each, a public card and two betting rounds. "
                 "Held out: MARSHAL never trains on it.",
    },
}

GAME_ORDER = list(GAMES)

# The chat template EnvManager installs (no tools): it keeps every message verbatim,
# including earlier <think> blocks, unlike Qwen3's stock template.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def chat_text(messages):
    """Render messages as EnvManager does, ready for generation."""
    parts = []
    if messages and messages[0]["role"] == "system":
        parts.append(f"{IM_START}system\n{messages[0]['content']}{IM_END}\n")
    else:
        parts.append(f"{IM_START}system\nYou are a helpful assistant.{IM_END}\n")
    for i, m in enumerate(messages):
        if m["role"] in ("user", "assistant") or (m["role"] == "system" and i > 0):
            parts.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    text = "".join(parts).replace(f"{IM_END}\n", IM_END)
    return text + "<think>\n"  # enable_think: force the model to think first


def parse_response(response):
    """EnvManager._parse_response with enable_think=True. Returns (think, answer, actions)."""
    match = re.search(r"^<think>(.*?)</think>\s*<answer>(.*?)</answer>$", response, re.DOTALL)
    if not match:
        return None, None, []
    think, answer = match.group(1), match.group(2)
    for token in SPECIAL_TOKENS:
        answer = answer.replace(token, "").strip()
        think = think.replace(token, "").strip()
    actions = [a.strip() for a in answer.split(ACTION_SEP) if a.strip()][:1]
    return think, answer, actions


def lenient_answer(text):
    """For players outside the training format (Claude): the last <answer>…</answer>."""
    found = re.findall(r"<answer>(.*?)</answer>", text or "", re.DOTALL)
    if not found:
        return []
    answer = found[-1]
    for token in SPECIAL_TOKENS:
        answer = answer.replace(token, "").strip()
    return [a.strip() for a in answer.split(ACTION_SEP) if a.strip()][:1]


def length_penalty(token_length):
    """EnvManager.compute_length_penalty: only responses past ~2048 tokens lose reward."""
    s = LENGTH_REWARD
    reward = s["coef"] - (token_length - s["min_len"]) / (s["max_len"] - s["min_len"])
    if reward > 0:
        reward *= s["upper"]
    if reward < 0:
        reward *= s["lower"]
    return reward


def jsonable(value):
    """numpy scalars and tuples from OpenSpiel/MARSHAL infos -> plain JSON values."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and value != value:
        return None
    return value


class GameStopped(Exception):
    pass


# ------------------------------------------------------------------- spectator views
CARD = "JQK"


def _grid(state_str):
    rows = [r for r in state_str.strip("\n").split("\n") if r.strip()]
    return [["X" if c == "x" else "O" if c == "o" else "" for c in row.strip()] for row in rows]


def _leduc(state):
    text = str(state)
    out = {"round": 1, "pot": 2, "money": [99, 99], "public": None, "cards": [None, None], "sequences": []}
    for line in text.split("\n"):
        if line.startswith("Round:"):
            out["round"] = int(line.split(":")[1])
        elif line.startswith("Pot:"):
            out["pot"] = int(line.split(":")[1])
        elif line.startswith("Money"):
            out["money"] = [int(x) for x in line.split(":")[1].split()]
        elif line.startswith("Cards"):
            ids = [int(x) for x in line.split(":")[1].split()]
            rank = lambda i: None if i < 0 else "JQK"[i // 2]  # noqa: E731
            out["public"] = rank(ids[0])
            out["cards"] = [rank(ids[1]), rank(ids[2])]
        elif line.startswith("Round ") and "sequence" in line:
            seq = line.split(":", 1)[1].strip()
            out["sequences"].append([s.strip() for s in seq.split(",") if s.strip()])
    # Chips each player put in: ante 1, raises of 2 (round 1) and 4 (round 2).
    bets = [1, 1]
    for rnd, seq in enumerate(out["sequences"][:2]):
        top = max(bets)
        for i, act in enumerate(seq):
            p = i % 2
            if act == "Raise":
                top += 2 if rnd == 0 else 4
                bets[p] = top
            elif act == "Call":
                bets[p] = top
    out["bets"] = bets
    out["pot"] = sum(bets)
    if state.is_terminal():
        out["returns"] = [float(r) for r in state.returns()]
    return out


def _hanabi(state):
    text = str(state)
    out = {"life": None, "info": None, "fireworks": {}, "deck": None, "discards": [], "hands": [[], []]}
    hands, current, in_hands, cur_seen = [[]], None, False, False
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("Life tokens:"):
            out["life"] = int(s.split(":")[1])
        elif s.startswith("Info tokens:"):
            out["info"] = int(s.split(":")[1])
        elif s.startswith("Fireworks:"):
            for fw in s.split(":", 1)[1].split():
                out["fireworks"][fw[0]] = int(fw[1:])
        elif s == "Hands:":
            in_hands = True
        elif s.startswith("Deck size:"):
            out["deck"] = int(s.split(":")[1])
            in_hands = False
        elif s.startswith("Discards:"):
            out["discards"] = s.split(":", 1)[1].split()
        elif in_hands and s == "-----":
            hands.append([])
        elif in_hands and s == "Cur player":
            current = len(hands) - 1
            cur_seen = True
        elif in_hands and s:
            card, _, knowledge = s.partition("||")
            known = knowledge.strip().split("|")[0] if knowledge else "XX"
            hands[-1].append({"card": card.strip(), "color_known": known[:1] != "X",
                              "rank_known": known[1:2] != "X"})
    out["hands"] = (hands + [[], []])[:2]
    out["current"] = current if cur_seen else None
    return out


def replay_states(state):
    """States after each player decision, rebuilt from OpenSpiel's history.

    Chance events that follow a decision (a public card, a Hanabi draw) are folded
    into that decision. Returns (state_before_first_decision, [state_after_each]).
    """
    s = state.get_game().new_initial_state()
    after = []
    for action in state.history():
        chance = s.is_chance_node()
        s.apply_action(action)
        if not chance:
            after.append(s.clone())
        elif after:
            after[-1] = s.clone()
    if not after:
        return s.clone(), []
    start = state.get_game().new_initial_state()
    for action in state.history():
        if not start.is_chance_node():
            break
        start.apply_action(action)
    return start, after


def spectator_view(game, state):
    """What the audience sees: the whole table, including private cards."""
    if state is None:
        return {}
    if game in ("tictactoe", "connect_four"):
        return {"type": "grid", "cells": _grid(str(state))}
    if game == "kuhn_poker":
        hist = state.history()
        cards = [CARD[hist[0]] if len(hist) > 0 else None, CARD[hist[1]] if len(hist) > 1 else None]
        bets = [1, 1]
        acts = []
        for i, a in enumerate(hist[2:]):
            acts.append({"player": i % 2, "action": "BET" if a == 1 else "PASS"})
            if a == 1:
                bets[i % 2] += 1
        view = {"type": "kuhn", "cards": cards, "bets": bets, "pot": sum(bets), "actions": acts}
        if state.is_terminal():
            view["returns"] = [float(r) for r in state.returns()]
        return view
    if game == "leduc_poker":
        return {"type": "leduc", **_leduc(state)}
    if game == "hanabi":
        return {"type": "hanabi", **_hanabi(state)}
    return {"type": "text", "text": str(state)}


# ---------------------------------------------------------------------------- game
class Game:
    """One game between two seats. Seats are players from display.players.

    At most one seat may be a bot: the game's own built-in opponent plays it
    (MCTS, CFR or Nash equilibrium), exactly as in MARSHAL's evaluation configs.
    """

    def __init__(self, game, seats, seed=0, sequence_length=DEFAULT_SEQUENCE_LENGTH,
                 max_new_tokens=DEFAULT_MAX_NEW_TOKENS, env_options=None, on_change=None,
                 ask_human=None, stop_event=None, source=None):
        from roll.agentic.env import REGISTERED_ENV_CONFIGS, REGISTERED_ENVS

        if game not in GAMES:
            raise ValueError(f"Unknown game {game!r}. Choose from {', '.join(GAMES)}.")
        if len(seats) != 2:
            raise ValueError("A MARSHAL game has two seats.")
        bots = [i for i, s in enumerate(seats) if s.kind == "bot"]
        if len(bots) > 1:
            raise ValueError("Only one seat can be the built-in bot.")
        if bots and not GAMES[game]["bot"]:
            raise ValueError(f"{GAMES[game]['title']} has no built-in opponent; both seats must play.")
        self.game = game
        self.seats = seats
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.max_new_tokens = int(max_new_tokens)
        self.on_change = on_change or (lambda: None)
        self.ask_human = ask_human
        self.stop_event = stop_event or threading.Event()
        options = dict(env_options or {})
        options["built_in_opponent"] = GAMES[game]["bot"] if bots else "none"
        if bots:
            options["opponent_player"] = bots[0]
        self.env_options = options
        self.env = REGISTERED_ENVS[game](REGISTERED_ENV_CONFIGS[game](**options))
        self.lock = threading.RLock()
        self.pending = None  # the move being thought about right now
        self.record = {
            "format": "marshal-game/1",
            "id": f"{game}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}",
            "game": game,
            "title": GAMES[game]["title"],
            "family": GAMES[game]["family"],
            "split": GAMES[game]["split"],
            "seed": self.seed,
            "env_options": jsonable(options),
            "settings": {"sequence_length": self.sequence_length, "max_new_tokens": self.max_new_tokens,
                         "format_penalty": FORMAT_PENALTY},
            "players": [self._seat_info(i, s) for i, s in enumerate(seats)],
            "prompts": {},
            "source": source or "live",
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished": None,
            "initial": None,
            "moves": [],
            "result": None,
            "turn_rewards": {"0": [], "1": []},
        }

    def _seat_info(self, i, seat):
        info = seat.describe()
        info["seat"] = i
        info["mark"] = GAMES[self.game]["marks"][i]
        if seat.kind == "bot":
            sims = self.env_options.get("max_simulations")
            label = GAMES[self.game]["bot_label"]
            if GAMES[self.game]["bot"] == "mcts":
                label = f"MCTS bot ({sims or 100} simulations)"
            info["name"] = label
        return info

    # ------------------------------------------------------------- bookkeeping
    @staticmethod
    def _update(history, actions_info=None, next_state=None):
        """EnvManager._update_cache_history."""
        if actions_info is not None:
            for k, v in actions_info.items():
                if k == "reward" and k in history[-1]:
                    history[-1][k] += v
                else:
                    history[-1][k] = v
        if next_state is not None:
            history.append(dict(next_state))

    def _views(self, results):
        """The board after each of these results (a forfeit leaves it unchanged)."""
        if all(not t["action"] for t in results):
            return [spectator_view(self.game, self.env.state)] * len(results)
        _, after = replay_states(self.env.state)
        views = [spectator_view(self.game, s) for s in after[-len(results):]]
        return [views[0]] * (len(results) - len(views)) + views

    def _log(self, results, current_player, format_reward=0.0, env_input=None, move_extra=None):
        """EnvManager._log_env_state, plus a move record for the display."""
        views = self._views(results)
        for idx, turn in enumerate(results):
            player = turn["current_player"]
            self.steps += 1
            if turn["done"]:
                self.done = True
            if self.steps >= MAX_ACTIONS_PER_TRAJ and not turn["done"]:
                self.done = self.truncated = True
            info = {
                "actions": turn["action"],
                "reward": turn["rewards"][player],
                "info": turn["info"],
            }
            next_state = {"player": turn["next_player"], "state": turn["observation"],
                          "legal_actions": turn["legal_actions"]}
            before = self.history[-1]
            if idx == 0 and env_input is not None:
                info["reward"] += format_reward + length_penalty(env_input["token_length"])
                info["llm_raw_response"] = env_input["llm_raw_response"]
            self._update(self.history, info, next_state)
            self._update(self.players_history[player], info, None)
            other = 1 - player
            if self.players_history[other]:
                self._update(self.players_history[other], {"reward": turn["rewards"][other], "info": turn["info"]})
            if not self.done:
                self._update(self.players_history[turn["next_player"]], None, next_state)

            move = {
                "n": len(self.record["moves"]) + 1,
                "player": player,
                "by": self.seats[player].kind,
                "action": turn["action"],
                "observation": before.get("state"),
                "legal": list((before.get("legal_actions") or {}).values()),
                "reward": info["reward"],
                "rewards": turn["rewards"],
                "done": bool(turn["done"]),
                "view": views[idx],
            }
            if idx == 0 and move_extra:
                move.update(move_extra)
            with self.lock:
                self.record["moves"].append(jsonable(move))
                self.pending = None
            if turn["done"]:
                self.final_info = turn["info"]
        self.on_change()

    # ----------------------------------------------------------- what players see
    def messages_for(self, player_id):
        """EnvManager._format_messages for one player, from the shared history."""
        prefix = self.env.get_prompt(mode="prefix", think=True, player_id=player_id)
        messages = [{"role": "system", "content": prefix["system"]},
                    {"role": "user", "content": prefix["user"]}]
        mode = self.env.include_opponent_turn
        for idx, content in enumerate(self.history):
            if messages[-1]["role"] != "user":
                messages.append({"role": "user", "content": ""})
            turn_idx = idx + 1
            opponent = content["player"] != player_id
            if opponent:
                if mode == "full":
                    head = (f"Information of Turn-{turn_idx}:\n\n"
                            "This is the other player's turn. "
                            "The game state, legal actions, as well as the chosen action of the other player "
                            "for this turn are provided below.\n\n")
                elif mode == "action_full":
                    head = (f"Information of Turn-{turn_idx}:\n\n"
                            "This is the other player's turn. The game state for this turn is not available to you. "
                            "The legal actions and chosen action of the other player for this turn are provided below.\n\n")
                elif mode == "action":
                    head = (f"Information of Turn-{turn_idx}:\n\n"
                            "This is the other player's turn. The game state for this turn is not available to you. "
                            "The chosen action of the other player for this turn is provided below.\n\n")
                else:
                    continue
            else:
                head = (f"Information of Turn-{turn_idx}:\n\n"
                        "This is your turn. The game state and legal actions for this turn are provided below. "
                        "Please choose your action and strictly follow the given output format in the response "
                        "instructions.\n\n")
            messages[-1]["content"] += head
            if "state" in content:
                legal = ", ".join(content["legal_actions"].values()) if content.get("legal_actions") else ""
                if opponent:
                    if mode == "full":
                        messages[-1]["content"] += f"GAME STATE:\n{content['state']}\n\nLEGAL ACTIONS:\n{legal}.\n\n"
                    elif mode == "action_full":
                        messages[-1]["content"] += f"LEGAL ACTIONS:\n{legal}.\n\n"
                    if len(content.get("actions", "")) > 0:
                        messages[-1]["content"] += f"CHOSEN ACTION:\n{content['actions']}\n"
                else:
                    messages[-1]["content"] += f"GAME STATE:\n{content['state']}\n\nLEGAL ACTIONS:\n{legal}.\n\n"
            if "llm_raw_response" in content and not opponent:
                messages.append({"role": "assistant", "content": content["llm_raw_response"]})
        return messages

    # -------------------------------------------------------------------- moves
    def _check_stop(self):
        if self.stop_event.is_set():
            raise GameStopped()

    def _set_pending(self, player, **fields):
        with self.lock:
            self.pending = {"player": player, "by": self.seats[player].kind, "started": time.time(),
                            "think": "", "text": "", **fields}
        self.on_change()

    def _partial(self, think, text):
        with self.lock:
            if self.pending is not None:
                self.pending["think"] = think
                self.pending["text"] = text
        self.on_change()

    def _model_turn(self, p, seat, legal):
        messages = self.messages_for(p)
        prompt = chat_text(messages)
        new_info = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
        rules = self.env.get_prompt(mode="prefix", think=True, player_id=p)["user"]
        if len(messages) == 2 and new_info.startswith(rules):
            new_info = new_info[len(rules):]  # the first message also carries the rules
        self._set_pending(p, legal=list(legal.values()), prompt_tail=new_info)
        started = time.time()
        if seat.kind == "claude":
            out = seat.move(messages, on_partial=self._partial, stop_event=self.stop_event)
            self._check_stop()
            actions = lenient_answer(out["text"])
            think = out.get("thinking") or out["text"].split("<answer>")[0].strip()
            answer = actions[0] if actions else None
            raw, token_length, prompt_tokens = out["text"], 0, None
            overlong_response = overlong_sequence = False
        else:
            prompt_tokens = seat.count_tokens(prompt)
            room = self.sequence_length - prompt_tokens
            if room <= 1:  # EnvManager.generate returns None: the episode is cut off
                self.done = self.truncated = True
                return None
            budget = max(min(self.max_new_tokens, room), 1)
            out = seat.generate(prompt, budget, on_partial=lambda t: self._partial("", t),
                                stop_event=self.stop_event)
            self._check_stop()
            raw = "<think>\n" + out["text"]
            think, answer, actions = parse_response(raw)
            token_length = out["completion_tokens"]
            token_left = self.sequence_length - prompt_tokens - token_length
            overlong_response = token_length >= OVERLONG_RESPONSE
            overlong_sequence = token_left <= MIN_TOKENS_LEFT
            if think is None:  # show what it wrote even though it doesn't count
                think = out["text"].split("</think>")[0]
        valid = [a for a in actions if a in legal.values()]
        illegal = [a for a in actions if a not in legal.values()]
        wrong_format = len(valid) != 1 or len(illegal) > 0
        forfeit = ("overlong_response" if overlong_response else "overlong_sequence" if overlong_sequence
                   else "format" if wrong_format else None)
        extra = {
            "think": think, "answer": answer, "raw": raw, "tokens": token_length,
            "prompt_tokens": prompt_tokens, "seconds": round(time.time() - started, 1),
            "valid": forfeit is None, "forfeit": forfeit, "prompt_tail": new_info,
        }
        if forfeit:
            results = self.env.get_losing_state(p, overlong_response, overlong_sequence)
            format_reward = min(FORMAT_PENALTY, 0)
        else:
            results = self.env.step(valid[0])
            format_reward = max(FORMAT_PENALTY, 0)
        env_input = {"token_length": token_length, "llm_raw_response": raw}
        self._log(results, p, format_reward, env_input, extra)
        return results

    def _human_turn(self, p, seat, legal):
        self._set_pending(p, legal=list(legal.values()))
        started = time.time()
        if self.ask_human is None:
            raise RuntimeError("A person is seated but nobody can take their move.")
        action = self.ask_human(p, list(legal.values()))
        self._check_stop()
        if action not in legal.values():
            raise RuntimeError(f"{action!r} is not a legal move.")
        results = self.env.step(action)
        extra = {"seconds": round(time.time() - started, 1), "valid": True, "forfeit": None}
        self._log(results, p, max(FORMAT_PENALTY, 0), {"token_length": 0, "llm_raw_response": action}, extra)

    # --------------------------------------------------------------------- run
    def play(self):
        """Play to the end and return the record. Raises GameStopped if stopped."""
        self.done = self.truncated = False
        self.steps = 0
        self.final_info = {}
        self.history = []
        self.players_history = {0: [], 1: []}
        for i in (0, 1):
            if self.seats[i].kind in ("model", "claude"):
                self.record["prompts"][str(i)] = self.env.get_prompt(mode="prefix", think=True, player_id=i)
        self_play = self.env.built_in_opponent == "none"
        current = 0 if self_play else 1 - self.env.opponent_player
        observation, opening = self.env.reset(seed=self.seed)
        first = {"player": 0, "state": observation["observation"], "legal_actions": observation["legal_actions"]}
        self._update(self.history, None, first)
        self._update(self.players_history[0], None, first)
        start, _ = replay_states(self.env.state)
        with self.lock:
            self.record["initial"] = jsonable({"observation": observation["observation"],
                                               "view": spectator_view(self.game, start)})
        self.on_change()
        if opening:
            self._log(opening, 0)
        while not self.done:
            self._check_stop()
            legal = self.history[-1]["legal_actions"]
            seat = self.seats[current]
            if seat.kind == "human":
                self._human_turn(current, seat, legal)
            elif seat.kind in ("model", "claude"):
                if self._model_turn(current, seat, legal) is None and self.done:
                    break
            else:
                raise RuntimeError(f"Seat {current} is a bot but it is not the bot's turn.")
            if self_play:
                current = self.env.current_player
        self._finish()
        return self.record

    def _finish(self):
        info = jsonable(self.final_info or {})
        result = {"info": info, "truncated": bool(self.truncated)}
        if self.game == "hanabi":
            score = info.get("player_0_return")
            result.update({"cooperative": True, "score": score, "winner": None})
        else:
            result["winner"] = info.get("winner")
            result["draw"] = bool(info.get("draw"))
            result["returns"] = [info.get("player_0_return"), info.get("player_1_return")]
        reason = "truncated" if self.truncated else "end"
        for p in (0, 1):
            for key, why in (("lose_for_overlong_response", "overlong_response"),
                             ("lose_for_overlong_sequence", "overlong_sequence"),
                             ("lose_for_wrong_format", "format")):
                if info.get(f"player_{p}_{key}"):
                    reason = f"forfeit_{why}"
                    result["forfeit_by"] = p
                    break
            if "forfeit_by" in result:
                break
        if reason == "forfeit_format" and result.get("forfeit_by") is not None:
            # get_losing_state flags wrong_format for every forfeit; keep the specific cause
            last = next((m for m in reversed(self.record["moves"]) if m.get("forfeit")), None)
            if last:
                reason = f"forfeit_{last['forfeit']}"
        result["reason"] = reason
        turn_rewards = {str(p): [round(float(h.get("reward", 0.0)), 4) for h in self.players_history[p]
                                 if "reward" in h] for p in (0, 1)}
        with self.lock:
            self.record["result"] = result
            self.record["turn_rewards"] = turn_rewards
            self.record["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.pending = None
        self.on_change()

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps({"record": self.record, "pending": self.pending}))
