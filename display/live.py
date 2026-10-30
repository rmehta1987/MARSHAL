"""Run one live MARSHAL game for the display.

Needs the venv with open_spiel (the games are MARSHAL's own classes). Every
screen open on the display follows the same game; a person can take a seat and
choose moves on the screen.
"""

import os
import threading
import time
import traceback

from display.config import Roster, load_config
from display.engine import GAMES, Game, GameStopped
from display.record import save_record


class LiveManager:
    def __init__(self, config_path=None, save_dir=None):
        self.cfg = load_config(config_path)
        self.roster = Roster(self.cfg)
        self.save_dir = save_dir
        self.game = None
        self.thread = None
        self.status = "idle"
        self.error = None
        self.awaiting = None
        self._human_move = None
        self.saved_path = None
        self.version = 0
        self.changed = threading.Condition()
        self.human_cond = threading.Condition()

    # ------------------------------------------------------------- info
    def options(self):
        defaults = self.cfg.get("defaults") or {}
        return {
            "enabled": True,
            "games": [{"key": k, "title": g["title"], "split": g["split"], "family": g["family"],
                       "bot": g["bot_label"]} for k, g in GAMES.items()],
            "players": self.roster.options(),
            "defaults": {"game": defaults.get("game", "tictactoe"),
                         "seats": list(defaults.get("seats") or ["human", "qwen3-4b"])},
            "limits": {"max_new_tokens": self.roster.max_new_tokens, "sequence_length": self.roster.sequence_length},
        }

    def _publish(self):
        with self.changed:
            self.version += 1
            self.changed.notify_all()

    def wait(self, last_version, timeout):
        with self.changed:
            if self.version == last_version:
                self.changed.wait(timeout)
            return self.version

    def snapshot(self):
        snap = {"status": self.status, "error": self.error, "awaiting": self.awaiting,
                "version": self.version, "now": time.time(), "record": None, "pending": None}
        if self.game is not None:
            snap.update(self.game.snapshot())
        return snap

    # ----------------------------------------------------------- control
    def start(self, request):
        game_key = request.get("game") or "tictactoe"
        if game_key not in GAMES:
            return f"Unknown game {game_key!r}."
        keys = list(request.get("seats") or [])
        if len(keys) != 2:
            return "Pick a player for both seats."
        try:
            seats = [self.roster.build(k) for k in keys]
        except ValueError as exc:
            return str(exc)
        kinds = [s.kind for s in seats]
        if kinds.count("bot") > 1:
            return "Only one seat can be the built-in bot."
        if "bot" in kinds and not GAMES[game_key]["bot"]:
            return f"{GAMES[game_key]['title']} has no built-in bot. Both seats must be players."
        checked = set()
        for key, seat in zip(keys, seats):
            if key in checked or not hasattr(seat, "check"):
                continue
            checked.add(key)
            problem = seat.check()
            if problem:
                return problem
        self.stop()
        seed = request.get("seed")
        seed = int(seed) if seed not in (None, "") else int(time.time()) % 100000
        self.stop_event = threading.Event()
        self.error = None
        self.awaiting = None
        self.saved_path = None
        self.game = Game(game_key, seats, seed=seed, max_new_tokens=self.roster.max_new_tokens,
                         sequence_length=self.roster.sequence_length, on_change=self._publish,
                         ask_human=self._ask_human, stop_event=self.stop_event, source="live")
        self.game.record["seat_keys"] = keys
        self.status = "running"
        self.thread = threading.Thread(target=self._run, args=(self.game,), daemon=True)
        self.thread.start()
        self._publish()
        return None

    def _run(self, game):
        try:
            record = game.play()
            if game is not self.game:
                return
            self.status = "finished"
            if self.save_dir:
                self.saved_path = save_record(record, self.save_dir)
                print(f"Saved live game: {self.saved_path}", flush=True)
        except GameStopped:
            if game is self.game:
                self.status = "stopped"
        except Exception as exc:  # noqa: BLE001 - show it on the screen
            traceback.print_exc()
            if game is self.game:
                self.status = "error"
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if game is self.game:
                self.awaiting = None
            self._publish()

    def stop(self):
        if self.game is not None and self.status == "running":
            self.stop_event.set()
            with self.human_cond:
                self.human_cond.notify_all()
            if self.thread is not None:
                self.thread.join(timeout=10)
            self.status = "stopped"
            self._publish()

    # ------------------------------------------------------------ people
    def _ask_human(self, player, legal):
        with self.human_cond:
            self._human_move = None
            self.awaiting = {"player": player, "legal": legal, "since": time.time()}
            self._publish()
            while self._human_move is None and not self.stop_event.is_set():
                self.human_cond.wait(1.0)
            move, self._human_move = self._human_move, None
            self.awaiting = None
        self._publish()
        return move

    def submit(self, player, action):
        with self.human_cond:
            if self.awaiting is None:
                return "Nobody is waiting for a move."
            if int(player) != self.awaiting["player"]:
                return "It isn't that seat's turn."
            if action not in self.awaiting["legal"]:
                return f"{action} isn't a legal move."
            self._human_move = action
            self.human_cond.notify_all()
        return None


def default_save_dir():
    return os.path.join("results", "display_games", "live")
