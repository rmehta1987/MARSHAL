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
            "attempts": {r: 0.0 for r in ROLES}, "fails": {r: 0 for r in ROLES}, "games": [],
        })
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
    return {"experiment": os.path.basename(os.path.normpath(results_dir)), "pairings": out,
            "models": {k: NAMES.get(k, k) for k in keys}}


if __name__ == "__main__":
    import sys
    print(json.dumps(summarize(*(sys.argv[1:2] or [DEFAULT_RESULTS])), indent=1))
