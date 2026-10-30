"""Summarize Decrypto games played by base Qwen3-4B and MARSHAL 4B.

Decrypto (../decrypto) is a word game about reading your partner and your
opponent, and MARSHAL never trained on it. display/decrypto/marshal_vs_qwen3_4b.yaml
runs its competitive setup with the two models; this module reads the results
Decrypto writes (results/<exp>/model_seed*/<enc>_<dec>_<int>_<modes>/env_seed*/stats.json)
and counts outcomes per pairing (team model × Eve model) for the Results screen.

Standard library only.
"""

import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "decrypto", "marshal_vs_qwen3_4b.yaml")
DEFAULT_RESULTS = os.path.join("..", "decrypto", "results", "marshal_vs_qwen3_4b")
NAMES = {"qwen3_4b": "Qwen3-4B", "marshal_4b": "MARSHAL 4B"}
ROLES = ("encoder", "decoder", "interceptor")


def model_keys(config_path=DEFAULT_CONFIG):
    try:
        with open(config_path, encoding="utf-8") as f:
            return re.findall(r"^\s*model_key:\s*(\S+)", f.read(), re.M)
    except OSError:
        return list(NAMES)


def split_combo(folder, keys):
    """'marshal_4b_marshal_4b_qwen3_4b_000_000' -> ('marshal_4b', 'marshal_4b', 'qwen3_4b')."""
    for enc in keys:
        for dec in keys:
            for itc in keys:
                if folder.startswith(f"{enc}_{dec}_{itc}_"):
                    return enc, dec, itc
    return None


def _common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def shares_stem(hint, keyword):
    """True when a clue is a near-copy of its keyword: edit -> editor, generate -> generation."""
    kw = re.sub(r"[^a-z]", "", keyword.lower())
    for word in re.findall(r"[a-z]+", hint.lower()):
        cp = _common_prefix(word, kw)
        if cp >= 4 and cp >= min(len(word), len(kw)) - 2:
            return True
    return False


def _last_guess(text):
    found = re.findall(r'"guess"\s*:\s*"(\d)\s*-\s*(\d)\s*-\s*(\d)"', text or "")
    return "-".join(found[-1]) if found else None


def episode_rounds(path):
    """Rounds of one game from Decrypto's combined_history.json: code, keywords, hints, guesses."""
    with open(path, encoding="utf-8") as f:
        history = json.load(f)
    rounds, cur = [], None
    for msg in history:
        text, role = msg.get("content") or "", msg.get("role")
        if role == "user" and text.startswith("[ENCODER]"):
            code = re.search(r"The code is (\d-\d-\d)", text)
            kws = re.search(r"The four keywords are:\s*\{(.*?)\}", text, re.S)
            if code and kws:
                cur = {"code": code.group(1), "hints": None, "decoder": None, "interceptor": None,
                       "keywords": dict(re.findall(r"(\d)\s*:\s*([^,\n}]+)", kws.group(1)))}
                cur["keywords"] = {k: v.strip() for k, v in cur["keywords"].items()}
                rounds.append(cur)
        elif cur is None:
            continue
        elif role == "user" and text.startswith("[DECODER]") and cur["hints"] is None:
            block = re.search(r"The hints given by the Encoder for this turn are:\s*\{(.*?)\}", text, re.S)
            if block:
                cur["hints"] = [h.strip() for _, h in re.findall(r"([abc])\s*:\s*([^,\n}]+)", block.group(1))]
        elif role == "assistant" and text.startswith("[DECODER]"):
            cur["decoder"] = _last_guess(text) or cur["decoder"]
        elif role == "assistant" and text.startswith("[INTERCEPTOR]"):
            cur["interceptor"] = _last_guess(text) or cur["interceptor"]
    return [r for r in rounds if r["hints"]]


def clue_stats(rounds):
    out = {"rounds": 0, "clues": 0, "stem_clues": 0, "rounds_with_stem": 0, "intercepts": 0,
           "intercepts_with_stem": 0, "miscomms": 0, "examples": []}
    for r in rounds:
        digits = r["code"].split("-")
        pairs = [(h, r["keywords"].get(d, "")) for h, d in zip(r["hints"], digits)]
        stems = [(h, k) for h, k in pairs if k and shares_stem(h, k)]
        out["rounds"] += 1
        out["clues"] += len(pairs)
        out["stem_clues"] += len(stems)
        intercepted = r["interceptor"] == r["code"]
        out["intercepts"] += intercepted
        out["miscomms"] += bool(r["decoder"]) and r["decoder"] != r["code"]
        if stems:
            out["rounds_with_stem"] += 1
            out["intercepts_with_stem"] += intercepted
            if len(out["examples"]) < 6:
                out["examples"] += [f"{k} → {h}" for h, k in stems][:2]
    return out


def _merge(a, b):
    for k, v in b.items():
        if k == "examples":
            a[k] = (a.get(k, []) + v)[:6]
        else:
            a[k] = a.get(k, 0) + v
    return a


def summarize(results_dir=DEFAULT_RESULTS, config_path=DEFAULT_CONFIG):
    """Outcome counts per (team model, Eve model), summed over env seeds."""
    if not results_dir or not os.path.isdir(results_dir):
        return None
    keys = model_keys(config_path)
    pairs = {}
    for path in sorted(glob.glob(os.path.join(results_dir, "model_seed*", "*", "env_seed*", "stats.json"))):
        combo = split_combo(os.path.basename(os.path.dirname(os.path.dirname(path))), keys)
        if not combo or combo[0] != combo[1]:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                s = json.load(f)
        except (OSError, ValueError):
            continue
        team, eve = combo[0], combo[2]
        p = pairs.setdefault((team, eve), {
            "team": team, "eve": eve, "team_name": NAMES.get(team, team), "eve_name": NAMES.get(eve, eve),
            "episodes": 0, "survived": 0, "intercepted": 0, "miscommunicated": 0, "both": 0, "_turns": 0.0,
            "attempts": {r: 0.0 for r in ROLES}, "fails": {r: 0 for r in ROLES}, "games": [], "clues": {},
        })
        for hist in sorted(glob.glob(os.path.join(os.path.dirname(path), "episode_*", "combined_history.json"))):
            try:
                _merge(p["clues"], clue_stats(episode_rounds(hist)))
            except (OSError, ValueError):
                pass
        n = int(s.get("survived", 0)) + int(s.get("intercept", 0)) + int(s.get("miscomm", 0)) + int(s.get("both", 0))
        p["episodes"] += n
        p["survived"] += int(s.get("survived", 0))
        p["intercepted"] += int(s.get("intercept", 0))
        p["miscommunicated"] += int(s.get("miscomm", 0))
        p["both"] += int(s.get("both", 0))
        p["_turns"] += float(s.get("avg_turns_per_episode", 0)) * n
        for r in ROLES:
            p["attempts"][r] += float(s.get(f"avg_{r}_attempts", 0)) * n
            p["fails"][r] += int(s.get(f"total_{r}_fails", 0))
        p["games"].append(os.path.relpath(os.path.dirname(path), results_dir))
    out = []
    for p in pairs.values():
        n = p["episodes"] or 1
        p["avg_rounds"] = round(p.pop("_turns") / n, 2)
        p["attempts"] = {r: round(v / n, 3) for r, v in p["attempts"].items()}
        out.append(p)
    order = {k: i for i, k in enumerate(keys)}
    out.sort(key=lambda p: (order.get(p["team"], 9), order.get(p["eve"], 9)))
    # How each model writes clues as Alice, over all its games (any Eve).
    clues = {}
    for p in out:
        c = clues.setdefault(p["team"], {"name": p["team_name"]})
        _merge(c, {k: v for k, v in p["clues"].items()})
    for c in clues.values():
        c.setdefault("clues", 0)
        c["stem_share"] = round(c.get("stem_clues", 0) / c["clues"], 3) if c["clues"] else None
        no_stem = c.get("rounds", 0) - c.get("rounds_with_stem", 0)
        c["intercept_rate_with_stem"] = (round(c["intercepts_with_stem"] / c["rounds_with_stem"], 3)
                                         if c.get("rounds_with_stem") else None)
        c["intercept_rate_without"] = (round((c.get("intercepts", 0) - c.get("intercepts_with_stem", 0)) / no_stem, 3)
                                       if no_stem else None)
    return {"experiment": os.path.basename(os.path.normpath(results_dir)), "pairings": out,
            "models": {k: NAMES.get(k, k) for k in keys}, "clues": clues}


SNAPSHOT = os.path.join(HERE, "data", "decrypto_summary.json")
CAP_TOKENS = 1015  # Decrypto's 1024-token thinking budget; capped moves measure 1022


def thinking_of(text):
    m = re.search(r"<think>(.*?)</think>", text or "", re.S)
    return m.group(1).strip() if m else None


def cap_stats(results_dir, count_tokens, config_path=DEFAULT_CONFIG):
    """How often each model's thinking ran into the 1024-token budget, per role.

    Decrypto doesn't record a truncation flag: stage 2 prefills </think> either way.
    A move whose thinking measures >= 1015 tokens hit the budget (cut off at 1022).
    """
    keys = model_keys(config_path)
    out = {}
    for combo_dir in sorted(glob.glob(os.path.join(results_dir, "model_seed*", "*"))):
        combo = split_combo(os.path.basename(combo_dir), keys)
        if not combo or combo[0] != combo[1]:
            continue
        for role, model in zip(ROLES, combo):
            for hist in glob.glob(os.path.join(combo_dir, "env_seed*", "episode_*", f"{role}_history.json")):
                try:
                    with open(hist, encoding="utf-8") as f:
                        messages = json.load(f)
                except (OSError, ValueError):
                    continue
                for m in messages:
                    think = thinking_of(m.get("content")) if m.get("role") == "assistant" else None
                    if think is None:
                        continue
                    n = count_tokens(think)
                    s = out.setdefault(model, {}).setdefault(role, {"moves": 0, "capped": 0, "tokens": 0})
                    s["moves"] += 1
                    s["capped"] += n >= CAP_TOKENS
                    s["tokens"] += n
    for roles in out.values():
        for s in roles.values():
            s["mean_tokens"] = round(s.pop("tokens") / s["moves"]) if s["moves"] else None
            s["cap_rate"] = round(s["capped"] / s["moves"], 3) if s["moves"] else None
    return out


def snapshot(results_dir=DEFAULT_RESULTS, out_path=SNAPSHOT, tokenize_url="http://localhost:8000"):
    """Save the summary plus thinking-budget counts (needs a vLLM /tokenize endpoint)."""
    import urllib.request

    def count(text):
        req = urllib.request.Request(tokenize_url.rstrip("/") + "/tokenize", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": "Qwen/Qwen3-4B", "prompt": text,
                                                      "add_special_tokens": False}).encode())
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(json.load(r)["count"])

    data = summarize(results_dir)
    if data is None:
        raise SystemExit(f"No Decrypto results in {results_dir}.")
    data["cap"] = cap_stats(results_dir, count)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return data


def load(results_dir=DEFAULT_RESULTS, snapshot_path=SNAPSHOT):
    """The live summary when the Decrypto results are here, else the committed snapshot.
    The thinking-budget counts come from the snapshot (they need a tokenizer)."""
    saved = None
    if snapshot_path and os.path.isfile(snapshot_path):
        with open(snapshot_path, encoding="utf-8") as f:
            saved = json.load(f)
    live = summarize(results_dir) if results_dir else None
    if live and live["pairings"]:
        if saved and saved.get("experiment") == live["experiment"]:
            live["cap"] = saved.get("cap")
        return live
    return saved


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if args[:1] == ["snapshot"]:
        d = snapshot(*(args[1:2] or [DEFAULT_RESULTS]))
        print(f"Wrote {SNAPSHOT}: {sum(p['episodes'] for p in d['pairings'])} games")
        print(json.dumps(d["cap"], indent=1))
    else:
        print(json.dumps(summarize(*(args[:1] or [DEFAULT_RESULTS])), indent=1))
