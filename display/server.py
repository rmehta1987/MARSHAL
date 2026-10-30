"""Serve the MARSHAL monitor display.

Replays and results (standard library only, no installs):
    python3 -m display.server
    # open http://localhost:8766 on the monitor, press F for full screen

Plus live games with local models, Claude or a person (needs the venv with open_spiel):
    .venv/bin/python -m display.server --live

Run from the repository root. See display/README.md.
"""

import argparse
import json
import mimetypes
import os
import sys
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from display.decrypto import DEFAULT_RESULTS as DECRYPTO_RESULTS
from display.decrypto import load as load_decrypto
from display.games import GameIndex, arena
from display.runs import RunIndex

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HERE, "static")
DEFAULT_GAMES = [os.path.join(HERE, "data", "games"), os.path.join("results", "display_games")]


def results_payload(runs, games, decrypto_dir=None):
    """Everything the Results screens need, in one object."""
    overview = runs.overview()
    records = games.records()
    overview["arena"] = arena(records, source="eval")
    overview["bringup_games"] = [g["id"] for g in games.list() if g.get("source") == "bringup-settings"]
    overview["decrypto"] = load_decrypto(decrypto_dir) if decrypto_dir is not None else None
    return overview


class Handler(BaseHTTPRequestHandler):
    games = None
    runs = None
    decrypto = None
    live = None
    live_problem = None
    quiet = True

    def log_message(self, fmt, *args):
        if not self.quiet:
            super().log_message(fmt, *args)

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > 100_000:
            raise ValueError("request too large")
        return json.loads(self.rfile.read(length) or b"{}")

    def _static(self, rel):
        path = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not path.startswith(STATIC_DIR + os.sep) or not os.path.isfile(path):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        with open(path, "rb") as f:
            body = f.read()
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".woff2"):
            ctype = "font/woff2"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                return self._static("index.html")
            if url.path.startswith("/static/"):
                return self._static(url.path[len("/static/"):])
            if url.path == "/api/games":
                self.games.scan()
                return self._json({"games": self.games.list()})
            if url.path == "/api/game":
                game = self.games.get(parse_qs(url.query).get("id", [""])[0])
                if game is None:
                    return self._json({"error": "No game with that id."}, HTTPStatus.NOT_FOUND)
                return self._json(game)
            if url.path == "/api/results":
                self.games.scan()
                self.runs.scan()
                return self._json(results_payload(self.runs, self.games, self.decrypto))
            if url.path == "/api/live/options":
                if self.live is None:
                    return self._json({"enabled": False, "reason": self.live_problem})
                return self._json(self.live.options())
            if url.path == "/api/live/stream":
                return self._stream()
            self.send_error(HTTPStatus.NOT_FOUND)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self):
        url = urlparse(self.path)
        if self.live is None:
            return self._json({"error": self.live_problem}, HTTPStatus.BAD_REQUEST)
        try:
            body = self._read_json()
            if url.path == "/api/live/start":
                problem = self.live.start(body)
            elif url.path == "/api/live/stop":
                self.live.stop()
                problem = None
            elif url.path == "/api/live/input":
                problem = self.live.submit(body.get("player"), body.get("action"))
            else:
                return self.send_error(HTTPStatus.NOT_FOUND)
            if problem:
                return self._json({"error": problem}, HTTPStatus.BAD_REQUEST)
            return self._json({"ok": True})
        except (ValueError, TypeError) as exc:
            return self._json({"error": f"Bad request: {exc}"}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            return self._json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def _stream(self):
        """Server-sent events: the whole game when a move lands, only the thinking in between."""
        if self.live is None:
            return self._json({"error": self.live_problem}, HTTPStatus.BAD_REQUEST)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last, last_key = -1, None
        while True:
            version = self.live.wait(last, timeout=15)
            if version == last:
                self.wfile.write(b": keep-alive\n\n")
            else:
                last = version
                snap = self.live.snapshot()
                rec = snap.get("record")
                key = (snap["status"], id(self.live.game), len(rec["moves"]) if rec else -1,
                       bool(rec and rec.get("result")), json.dumps(snap.get("awaiting")))
                if key == last_key:
                    snap = {k: snap[k] for k in ("status", "pending", "awaiting", "version", "now", "error")}
                    snap["partial"] = True
                last_key = key
                self.wfile.write(f"data: {json.dumps(snap)}\n\n".encode())
                time.sleep(0.12)  # let streamed tokens pile up between updates
            self.wfile.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES,
                        help="Folders with *.marshal-game.json files. Default: display/data/games and "
                             "results/display_games")
    parser.add_argument("--live", action="store_true", help="Offer live games (needs the venv with open_spiel).")
    parser.add_argument("--config", default=None, help="Player config for live games (default display/strix_halo.yaml).")
    parser.add_argument("--save-live", default=os.path.join("results", "display_games", "live"), metavar="DIR",
                        help="Where finished live games are saved; they join the replays. '' keeps none.")
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 lets other machines open the display (and start live games).")
    parser.add_argument("--decrypto", default=DECRYPTO_RESULTS, metavar="DIR",
                        help="Decrypto results with MARSHAL in it (display/decrypto/README). Default: "
                             f"{DECRYPTO_RESULTS}; '' hides the Decrypto page.")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    roots = list(args.games)
    if args.live and args.save_live and not any(os.path.abspath(args.save_live).startswith(os.path.abspath(r))
                                                for r in roots):
        roots.append(args.save_live)
    Handler.games = GameIndex(roots)
    found = len(Handler.games.scan())
    Handler.runs = RunIndex(".")
    Handler.decrypto = args.decrypto if args.decrypto else None
    Handler.quiet = not args.verbose
    if args.live:
        try:
            from display.live import LiveManager
            import pyspiel  # noqa: F401 - fail now, not on the first game
        except ImportError as exc:
            raise SystemExit(f"Live games need open_spiel and PyYAML ({exc}). See display/README.md.") from exc
        Handler.live = LiveManager(args.config, save_dir=args.save_live or None)
        names = ", ".join(p["name"] for p in Handler.live.roster.options())
        print(f"Live games on: {names}", flush=True)
    else:
        Handler.live_problem = "Start the server with --live (in the venv with open_spiel) to play live games."

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"Found {found} saved games and {len(Handler.runs.runs)} training runs.", flush=True)
    print(f"MARSHAL display: http://{host}:{args.port}   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if Handler.live:
            Handler.live.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
