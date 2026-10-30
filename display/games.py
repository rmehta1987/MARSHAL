"""Saved MARSHAL games: find them, list them, and count the Arena results.

A game file is one `*.marshal-game.json` written by display.record or by a
live game (see engine.Game.record). Only the standard library is used, so
replays need no installs.
"""

import glob
import json
import math
import os

GAME_ORDER = ["tictactoe", "kuhn_poker", "hanabi", "connect_four", "leduc_poker"]


class GameIndex:
    def __init__(self, roots):
        self.roots = [r for r in roots if r]
        self._cache = {}  # path -> (mtime, record)

    def scan(self):
        seen = set()
        for root in self.roots:
            if os.path.isfile(root) and root.endswith(".marshal-game.json"):
                paths = [root]
            else:
                paths = glob.glob(os.path.join(root, "**", "*.marshal-game.json"), recursive=True)
            for path in paths:
                path = os.path.abspath(path)
                seen.add(path)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                cached = self._cache.get(path)
                if cached and cached[0] == mtime:
                    continue
                try:
                    with open(path, encoding="utf-8") as f:
                        record = json.load(f)
                except (OSError, ValueError):
                    continue
                if record.get("format") != "marshal-game/1" or not record.get("result"):
                    continue
                self._cache[path] = (mtime, record)
        for path in list(self._cache):
            if path not in seen:
                del self._cache[path]
        return self.records()

    def records(self):
        recs = [rec for _, rec in self._cache.values()]
        recs.sort(key=lambda r: (GAME_ORDER.index(r["game"]) if r["game"] in GAME_ORDER else 99,
                                 r.get("started") or "", r["id"]))
        return recs

    def get(self, game_id):
        for _, rec in self._cache.values():
            if rec["id"] == game_id:
                return rec
        return None

    def list(self):
        return [summary(r) for r in self.records()]


def outcome_for(record, seat):
    """'win' | 'draw' | 'loss' | 'forfeit' | 'other' from one seat's point of view."""
    r = record.get("result") or {}
    if r.get("reason", "").startswith("forfeit") and r.get("forfeit_by") == seat:
        return "forfeit"
    if r.get("truncated"):
        return "other"
    if r.get("cooperative"):
        return "other"
    if r.get("draw"):
        return "draw"
    if r.get("winner") == seat:
        return "win"
    if r.get("winner") in (0, 1):
        return "loss"
    return "other"


def summary(record):
    r = record.get("result") or {}
    players = [{k: p.get(k) for k in ("name", "kind", "model", "mark")} for p in record["players"]]
    model_moves = [m for m in record["moves"] if m.get("by") in ("model", "claude")]
    return {
        "id": record["id"], "game": record["game"], "title": record["title"], "split": record.get("split"),
        "players": players, "source": record.get("source"), "started": record.get("started"),
        "moves": len(record["moves"]), "seed": record.get("seed"),
        "settings": record.get("settings"),
        "result": {k: r.get(k) for k in ("winner", "draw", "reason", "forfeit_by", "score", "cooperative", "returns")},
        "tokens": sum(m.get("tokens") or 0 for m in model_moves),
    }


def _wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def arena(records, source="eval"):
    """Count outcomes per (game, model) for games against the built-in bots,
    plus Hanabi self-play scores and model-vs-model head-to-heads."""
    cells, h2h = {}, {}
    for rec in records:
        if source and rec.get("source") != source:
            continue
        seats = rec["players"]
        kinds = [p["kind"] for p in seats]
        game = rec["game"]
        if game == "hanabi" and kinds == ["model", "model"] and seats[0].get("model") == seats[1].get("model"):
            key = (game, seats[0].get("model"))
            cell = cells.setdefault(key, _new_cell(game, seats[0], "self-play partner (same model)"))
            score = (rec["result"] or {}).get("score")
            cell["games"] += 1
            if any(outcome_for(rec, s) == "forfeit" for s in (0, 1)):
                cell["outcomes"]["forfeit"] += 1
            else:
                cell["outcomes"]["other"] += 1  # played out; the score says how well
                cell["scores"].append(score if score is not None else 0.0)
            cell["ids"].append(rec["id"])
            _tokens(cell, rec, (0, 1))
            continue
        if "bot" in kinds and kinds.count("model") == 1:
            seat = kinds.index("model")
            key = (game, seats[seat].get("model"))
            cell = cells.setdefault(key, _new_cell(game, seats[seat], seats[1 - seat]["name"]))
            outcome = outcome_for(rec, seat)
            cell["games"] += 1
            cell["outcomes"][outcome] = cell["outcomes"].get(outcome, 0) + 1
            cell["by_seat"][str(seat)][outcome] = cell["by_seat"][str(seat)].get(outcome, 0) + 1
            returns = (rec["result"] or {}).get("returns") or [None, None]
            if returns[seat] is not None:
                cell["returns"].append(returns[seat])
            cell["ids"].append(rec["id"])
            _tokens(cell, rec, (seat,))
            continue
        if kinds == ["model", "model"] and seats[0].get("model") != seats[1].get("model"):
            pair = tuple(sorted([seats[0].get("model"), seats[1].get("model")]))
            hk = (game,) + pair
            cell = h2h.setdefault(hk, {"game": game, "title": rec["title"], "models": list(pair), "games": 0,
                                       "wins": {pair[0]: 0, pair[1]: 0}, "draws": 0, "forfeits": {pair[0]: 0, pair[1]: 0},
                                       "names": {}, "ids": []})
            cell["games"] += 1
            cell["ids"].append(rec["id"])
            for s in (0, 1):
                cell["names"][seats[s].get("model")] = seats[s]["name"]
            for s in (0, 1):
                o = outcome_for(rec, s)
                if o == "win":
                    cell["wins"][seats[s].get("model")] += 1
                elif o == "forfeit":
                    cell["forfeits"][seats[s].get("model")] += 1
            if (rec["result"] or {}).get("draw"):
                cell["draws"] += 1
    out = []
    for (game, model), cell in cells.items():
        n = cell["games"]
        o = cell["outcomes"]
        cell["win_rate"] = round(o.get("win", 0) / n, 3) if n else None
        cell["win_ci"] = _wilson(o.get("win", 0), n)
        cell["forfeit_rate"] = round(o.get("forfeit", 0) / n, 3) if n else None
        cell["mean_return"] = round(sum(cell["returns"]) / len(cell["returns"]), 3) if cell["returns"] else None
        cell["mean_score"] = round(sum(cell["scores"]) / len(cell["scores"]), 3) if cell["scores"] else None
        t = cell.pop("_tok")
        cell["tokens_per_answer"] = round(sum(t) / len(t)) if t else None
        cell["answers"] = len(t)
        out.append(cell)
    out.sort(key=lambda c: (GAME_ORDER.index(c["game"]) if c["game"] in GAME_ORDER else 99, c["model"] or ""))
    return {"cells": out, "head_to_head": list(h2h.values())}


def _new_cell(game, player, opponent):
    return {"game": game, "model": player.get("model"), "name": player.get("name"), "opponent": opponent,
            "games": 0, "outcomes": {"win": 0, "draw": 0, "loss": 0, "forfeit": 0, "other": 0},
            "by_seat": {"0": {}, "1": {}}, "returns": [], "scores": [], "ids": [], "_tok": []}


def _tokens(cell, rec, seats):
    for m in rec["moves"]:
        if m.get("player") in seats and m.get("by") == "model" and m.get("tokens"):
            cell["_tok"].append(m["tokens"])
