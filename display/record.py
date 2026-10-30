"""Play and save MARSHAL games without the screen, for replays and evaluation.

Examples (from the repo root, in the venv that has open_spiel):

    # 10 Connect Four games, MARSHAL 4B against the MCTS bot, 4 at a time,
    # MARSHAL moving first in half of them
    .venv/bin/python -m display.record --game connect_four --seats marshal-4b bot \
        --games 10 --swap --parallel 4

    # Base Qwen3-4B against MARSHAL 4B at Tic-Tac-Toe
    .venv/bin/python -m display.record --game tictactoe --seats qwen3-4b marshal-4b --games 2 --swap

Games are saved one file each (`<id>.marshal-game.json`) under --out, where
the display finds them. Players are the keys in the config (default
display/strix_halo.yaml); `bot` is the game's built-in opponent.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from display.config import Roster, load_config
from display.engine import GAMES, Game

DEFAULT_OUT = os.path.join("results", "display_games")


def save_record(record, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{record['id']}.marshal-game.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def describe_result(record):
    r = record.get("result") or {}
    names = [p["name"] for p in record["players"]]
    if r.get("cooperative"):
        return f"score {r.get('score')}" + (f" ({r['reason']})" if r.get("reason", "end") != "end" else "")
    if r.get("reason", "").startswith("forfeit"):
        return f"{names[r['forfeit_by']]} forfeits ({r['reason'][8:]})"
    if r.get("draw"):
        return "draw"
    if r.get("winner") in (0, 1):
        return f"{names[r['winner']]} wins"
    return r.get("reason", "?")


def parse_options(pairs):
    out = {}
    for pair in pairs or []:
        k, _, v = pair.partition("=")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--game", required=True, choices=list(GAMES))
    parser.add_argument("--seats", nargs=2, required=True, metavar=("SEAT0", "SEAT1"),
                        help="Player keys from the config, first mover first.")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first game; game i uses seed+i.")
    parser.add_argument("--swap", action="store_true", help="Swap the seats every other game.")
    parser.add_argument("--parallel", type=int, default=1, help="Games played at once.")
    parser.add_argument("--config", default=None, help="Player config (default display/strix_halo.yaml).")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--tag", default="recorded", help="Stored as the game's source, e.g. eval.")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--env", nargs="*", metavar="KEY=VALUE", help="Game options, e.g. max_simulations=1000.")
    args = parser.parse_args(argv)

    roster = Roster(load_config(args.config))
    for key in set(args.seats):
        if key in ("human",):
            parser.error("display.record can't seat a person; use the display for that.")
        problem = getattr(roster.build(key), "check", lambda: None)()
        if problem:
            print(problem, file=sys.stderr)
            return 1
    env_options = parse_options(args.env)
    print_lock = threading.Lock()

    def play(i):
        seats = list(args.seats)
        if args.swap and i % 2 == 1:
            seats.reverse()
        game = Game(args.game, [roster.build(k) for k in seats], seed=args.seed + i,
                    max_new_tokens=args.max_new_tokens or roster.max_new_tokens,
                    sequence_length=args.sequence_length or roster.sequence_length,
                    env_options=env_options, source=args.tag)
        game.record["seat_keys"] = seats
        t = time.time()
        record = game.play()
        path = save_record(record, args.out)
        with print_lock:
            print(f"[{i + 1}/{args.games}] seed {args.seed + i} {seats[0]} vs {seats[1]}: "
                  f"{describe_result(record)} · {len(record['moves'])} moves · {time.time() - t:.0f}s · {path}",
                  flush=True)
        return record

    failures = 0
    with ThreadPoolExecutor(max(1, args.parallel)) as pool:
        futures = [pool.submit(play, i) for i in range(args.games)]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failures += 1
                with print_lock:
                    print(f"A game failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
