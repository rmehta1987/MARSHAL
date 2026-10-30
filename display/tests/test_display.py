"""Tests for the MARSHAL display.

    python3 -m unittest discover -s display/tests -t .        # replay side, standard library only
    .venv/bin/python -m unittest discover -s display/tests -t .   # everything (needs open_spiel)
"""

import json
import os
import random
import re
import struct
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from display import engine
from display.games import GameIndex, arena, outcome_for
from display.runs import RunIndex, keep_key, parse_log, to_series
from display.tfevents import read_scalars

try:
    import pyspiel  # noqa: F401

    HAVE_SPIEL = True
except ImportError:
    HAVE_SPIEL = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ScriptedModel:
    """Stands in for a vLLM model: picks a legal move from the prompt."""

    kind = "model"

    def __init__(self, name="Scripted", bad_on_call=None, rng=None, tokens=50):
        self.name = name
        self.bad_on_call = bad_on_call
        self.calls = 0
        self.rng = rng or random.Random(0)
        self.tokens = tokens
        self.prompts = []

    def describe(self):
        return {"kind": "model", "name": self.name, "model": f"test/{self.name}"}

    def count_tokens(self, text):
        return len(text) // 4

    def generate(self, prompt, max_tokens, on_partial=None, stop_event=None):
        self.calls += 1
        self.prompts.append(prompt)
        legal = re.findall(r"LEGAL ACTIONS:\n(.*?)\.\n", prompt)[-1].split(", ")
        pick = self.rng.choice(legal)
        text = f"Thinking about {pick}.\n</think>\n<answer>{pick}</answer>"
        if self.bad_on_call == self.calls:
            text = "I will just say it: " + pick
        if on_partial:
            on_partial(text[:5])
        return {"text": text, "completion_tokens": self.tokens, "prompt_tokens": None}


# ----------------------------------------------------------------- engine rules
class EngineRulesTest(unittest.TestCase):
    def test_chat_text_matches_envmanager_template(self):
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "<think>\nA\n</think><answer>x</answer>"},
                {"role": "user", "content": "U2"}]
        text = engine.chat_text(msgs)
        self.assertEqual(text, "<|im_start|>system\nSYS<|im_end|><|im_start|>user\nU1<|im_end|>"
                               "<|im_start|>assistant\n<think>\nA\n</think><answer>x</answer><|im_end|>"
                               "<|im_start|>user\nU2<|im_end|><|im_start|>assistant\n<think>\n")

    def test_parse_response_is_strict(self):
        think, answer, actions = engine.parse_response("<think>\nhmm\n</think>\n<answer><X(1,1)></answer>")
        self.assertEqual((think, answer, actions), ("hmm", "<X(1,1)>", ["<X(1,1)>"]))
        self.assertEqual(engine.parse_response("<think>\nhmm\n</think><answer>a</answer> extra")[2], [])
        self.assertEqual(engine.parse_response("<think>\nno close tag <answer>a</answer>")[2], [])
        self.assertEqual(engine.parse_response("<think>x</think><answer>a || b</answer>")[2], ["a"])

    def test_lenient_answer_takes_the_last(self):
        self.assertEqual(engine.lenient_answer("I think <answer>a</answer> no, <answer><BET></answer>"), ["<BET>"])
        self.assertEqual(engine.lenient_answer("no tags"), [])

    def test_length_penalty(self):
        self.assertEqual(engine.length_penalty(500), 0.0)
        self.assertEqual(engine.length_penalty(2048), 0.0)
        self.assertAlmostEqual(engine.length_penalty(4096), 0.5 * (1 - (4096 - 11) / 2037))

    def test_jsonable(self):
        class Num:
            def item(self):
                return 3
        self.assertEqual(engine.jsonable({"a": (Num(), float("nan"))}), {"a": [3, None]})


@unittest.skipUnless(HAVE_SPIEL, "needs open_spiel (use the venv)")
class EngineGameTest(unittest.TestCase):
    def play(self, game, seats, seed=3, **kw):
        g = engine.Game(game, seats, seed=seed, **kw)
        rec = g.play()
        json.dumps(rec)
        return g, rec

    def test_self_play_tictactoe(self):
        a, b = ScriptedModel("A"), ScriptedModel("B")
        _, rec = self.play("tictactoe", [a, b])
        self.assertTrue(rec["result"])
        n0 = sum(1 for m in rec["moves"] if m["player"] == 0)
        self.assertEqual(len(rec["turn_rewards"]["0"]), n0)
        for m in rec["moves"]:
            self.assertTrue(m["valid"])
            self.assertEqual(m["view"]["type"], "grid")
        # each move adds exactly one mark
        counts = [sum(c != "" for row in m["view"]["cells"] for c in row) for m in rec["moves"]]
        self.assertEqual(counts, list(range(1, len(counts) + 1)))
        # second player's prompt shows the first player's chosen action (include_opponent_turn=full)
        self.assertIn("CHOSEN ACTION:\n<X(", b.prompts[0])
        self.assertTrue(b.prompts[0].endswith("<|im_start|>assistant\n<think>\n"))

    def test_turn_rewards_follow_envmanager(self):
        _, rec = self.play("tictactoe", [ScriptedModel("A", rng=random.Random(1)), ScriptedModel("B", rng=random.Random(2))])
        r = rec["result"]
        tr = rec["turn_rewards"]
        if r.get("draw"):
            self.assertAlmostEqual(tr["0"][-1], 0.05 + 0.1, places=4)
        else:
            w = r["winner"]
            self.assertAlmostEqual(tr[str(w)][-1], 1.05, places=4)
            self.assertAlmostEqual(tr[str(1 - w)][-1], 0.05 - 1, places=4)
        for p in "01":
            for v in tr[p][:-1]:
                self.assertAlmostEqual(v, 0.05, places=4)

    def test_format_forfeit(self):
        _, rec = self.play("tictactoe", [ScriptedModel("A"), ScriptedModel("B", bad_on_call=1)])
        self.assertEqual(rec["result"]["reason"], "forfeit_format")
        self.assertEqual(rec["result"]["forfeit_by"], 1)
        self.assertEqual(rec["result"]["winner"], 0)
        self.assertEqual(rec["turn_rewards"]["1"], [-11.0])
        self.assertEqual(rec["moves"][-1]["forfeit"], "format")

    def test_overlong_response_forfeit(self):
        _, rec = self.play("kuhn_poker", [ScriptedModel("A", tokens=4096), engine_bot()])
        self.assertEqual(rec["result"]["reason"], "forfeit_overlong_response")

    def test_bot_opens_and_views_replay_each_move(self):
        _, rec = self.play("tictactoe", [engine_bot(), ScriptedModel("B")])
        self.assertEqual(rec["moves"][0]["by"], "bot")
        self.assertEqual(sum(c != "" for row in rec["initial"]["view"]["cells"] for c in row), 0)
        counts = [sum(c != "" for row in m["view"]["cells"] for c in row) for m in rec["moves"] if m["action"]]
        self.assertEqual(counts, list(range(1, len(counts) + 1)))
        self.assertEqual(rec["players"][0]["name"], "MCTS bot (100 simulations)")

    def test_kuhn_and_leduc_views(self):
        _, rec = self.play("kuhn_poker", [ScriptedModel("A"), engine_bot()], seed=4)
        self.assertEqual(rec["initial"]["view"]["cards"], ["K", "J"])
        self.assertIn("returns", rec["moves"][-1]["view"])
        _, rec = self.play("leduc_poker", [ScriptedModel("A"), engine_bot()], seed=4)
        last = rec["moves"][-1]["view"]
        self.assertEqual(sum(last["bets"]), last["pot"])
        self.assertTrue(all(b >= 1 for b in last["bets"]))

    def test_hanabi_is_cooperative(self):
        _, rec = self.play("hanabi", [ScriptedModel("A"), ScriptedModel("B")])
        self.assertTrue(rec["result"]["cooperative"])
        self.assertEqual(len(rec["initial"]["view"]["hands"]), 2)
        with self.assertRaises(ValueError):
            engine.Game("hanabi", [ScriptedModel(), engine_bot()])

    def test_only_one_bot(self):
        with self.assertRaises(ValueError):
            engine.Game("tictactoe", [engine_bot(), engine_bot()])

    def test_connect_four_keeps_marshal_action_strings(self):
        _, rec = self.play("connect_four", [ScriptedModel("A"), ScriptedModel("B")])
        # MARSHAL's Connect Four env wraps a model's move string a second time; keep it as the models saw it
        self.assertRegex(rec["moves"][0]["action"], r"^<X\(<X\(\d\)>\)>$")


def engine_bot():
    from display.players import BotPlayer
    return BotPlayer()


# ------------------------------------------------------------------ results data
class RunsTest(unittest.TestCase):
    def test_parse_log_json_and_trimmed_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.out")
            with open(path, "w") as f:
                f.write('\x1b[36m[INFO] {"system/step": 0, "critic/score/mean": -11.0, "memory/a": 1, '
                        '"env/T/response_length_player_0_turn_0": 3}\n')
                f.write("step  1: samples=32  tps=260.1  score=-5.475  grad_norm=0.5797\n")
            steps = parse_log(path)
        self.assertEqual(steps[0], {"system/step": 0, "critic/score/mean": -11.0})
        self.assertEqual(steps[1]["critic/score/mean"], -5.475)
        self.assertEqual(steps[1]["actor_train/grad_norm"], 0.5797)
        self.assertEqual(to_series(steps)["critic/score/mean"], [[0, -11.0], [1, -5.475]])
        self.assertFalse(keep_key("time/foo"))
        self.assertTrue(keep_key("time/actor_train/train_step/total"))

    def test_tfevents_reader(self):
        def varint(n):
            out = b""
            while True:
                b = n & 0x7F
                n >>= 7
                out += bytes([b | (0x80 if n else 0)])
                if not n:
                    return out

        def field(num, wire, payload):
            key = varint((num << 3) | wire)
            return key + (varint(len(payload)) + payload if wire == 2 else payload)

        def event(step, tag, value):
            val = field(1, 2, tag.encode()) + field(2, 5, struct.pack("<f", value))
            summary = field(1, 2, val)
            return field(2, 0, varint(step)) + field(5, 2, summary)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "events.out.tfevents.1")
            with open(path, "wb") as f:
                for step, v in [(0, -5.5), (1, -1.25)]:
                    data = event(step, "critic/score/mean", v)
                    f.write(struct.pack("<Q", len(data)) + b"\0" * 4 + data + b"\0" * 4)
            scalars = read_scalars(path)
        self.assertEqual(scalars, {"critic/score/mean": {0: -5.5, 1: -1.25}})

    def test_ledger_loads_midway_series(self):
        index = RunIndex(ROOT)
        run = index.get("midway-scaleup-50259767")
        self.assertEqual(run["points"], 20)
        pg = run["series"]["actor/pg_loss"]
        self.assertEqual(sum(1 for _, v in pg if v == 0), 14)
        forfeits = run["series"]["env/TicTacToe/player_1_lose_for_wrong_format"]
        self.assertTrue(all(v == 1.0 for _, v in forfeits))
        self.assertTrue(index.get("polaris-400step-7198659")["milestones"])


def fake_record(game, seats, result, source="eval", gid=None):
    return {"format": "marshal-game/1", "id": gid or f"{game}-{random.random()}", "game": game, "title": game,
            "source": source, "players": seats, "moves": [{"player": 0, "by": "model", "tokens": 100}],
            "result": result}


class ArenaTest(unittest.TestCase):
    def test_counts_from_the_models_side(self):
        m = {"kind": "model", "name": "M", "model": "x/M"}
        b = {"kind": "bot", "name": "MCTS bot (100 simulations)"}
        recs = [
            fake_record("tictactoe", [m, b], {"winner": 0, "draw": False, "returns": [1, -1], "reason": "end"}),
            fake_record("tictactoe", [b, m], {"winner": 0, "draw": False, "returns": [1, -1], "reason": "end"}),
            fake_record("tictactoe", [b, m], {"winner": 0, "draw": False, "reason": "forfeit_format", "forfeit_by": 1,
                                              "returns": [1, -1]}),
            fake_record("tictactoe", [m, b], {"winner": -1, "draw": True, "returns": [0, 0], "reason": "end"}),
            fake_record("tictactoe", [m, b], {"winner": 0, "draw": False, "reason": "end"}, source="live"),
        ]
        cells = arena(recs)["cells"]
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["outcomes"], {"win": 1, "draw": 1, "loss": 1, "forfeit": 1, "other": 0})
        self.assertEqual(cells[0]["games"], 4)
        self.assertEqual(outcome_for(recs[2], 1), "forfeit")

    def test_hanabi_self_play_scores(self):
        m = {"kind": "model", "name": "M", "model": "x/M"}
        recs = [fake_record("hanabi", [m, m], {"cooperative": True, "score": 3.0, "reason": "end"}),
                fake_record("hanabi", [m, m], {"cooperative": True, "score": 1.0, "reason": "forfeit_format",
                                               "forfeit_by": 0})]
        cell = arena(recs)["cells"][0]
        self.assertEqual(cell["mean_score"], 3.0)
        self.assertEqual(cell["outcomes"]["forfeit"], 1)

    def test_index_reads_saved_games(self):
        with tempfile.TemporaryDirectory() as d:
            rec = fake_record("tictactoe", [{"kind": "model", "name": "M"}, {"kind": "bot", "name": "B"}],
                              {"winner": 0, "reason": "end"}, gid="g1")
            with open(os.path.join(d, "g1.marshal-game.json"), "w") as f:
                json.dump(rec, f)
            with open(os.path.join(d, "broken.marshal-game.json"), "w") as f:
                f.write("{")
            index = GameIndex([d])
            self.assertEqual([g["id"] for g in index.list()], [])  # list before scan is empty
            index.scan()
            self.assertEqual([g["id"] for g in index.list()], ["g1"])
            self.assertEqual(index.get("g1")["title"], "tictactoe")


class DecryptoSummaryTest(unittest.TestCase):
    def test_counts_per_pairing(self):
        from display.decrypto import split_combo, summarize
        keys = ["qwen3_4b", "marshal_4b"]
        self.assertEqual(split_combo("marshal_4b_marshal_4b_qwen3_4b_000_000", keys),
                         ("marshal_4b", "marshal_4b", "qwen3_4b"))
        with tempfile.TemporaryDirectory() as d:
            for seed, stats in ((0, {"intercept": 1, "miscomm": 1, "both": 0, "survived": 2, "avg_turns_per_episode": 6.0,
                                     "avg_encoder_attempts": 0.5, "total_decoder_fails": 1}),
                                (1, {"intercept": 0, "miscomm": 0, "both": 1, "survived": 3, "avg_turns_per_episode": 8.0})):
                folder = os.path.join(d, "model_seed0", "marshal_4b_marshal_4b_qwen3_4b_000_000", f"env_seed{seed}")
                os.makedirs(folder)
                with open(os.path.join(folder, "stats.json"), "w") as f:
                    json.dump(stats, f)
            # an encoder != decoder combo is not a team pairing and is skipped
            os.makedirs(os.path.join(d, "model_seed0", "qwen3_4b_marshal_4b_qwen3_4b_000_000", "env_seed0"))
            with open(os.path.join(d, "model_seed0", "qwen3_4b_marshal_4b_qwen3_4b_000_000", "env_seed0", "stats.json"), "w") as f:
                json.dump({"survived": 4}, f)
            out = summarize(d)
        self.assertEqual(len(out["pairings"]), 1)
        p = out["pairings"][0]
        self.assertEqual((p["team_name"], p["eve_name"]), ("MARSHAL 4B", "Qwen3-4B"))
        self.assertEqual((p["episodes"], p["survived"], p["intercepted"], p["miscommunicated"], p["both"]), (8, 5, 1, 1, 1))
        self.assertEqual(p["avg_rounds"], 7.0)
        self.assertEqual(p["attempts"]["encoder"], 0.25)
        self.assertEqual(p["fails"]["decoder"], 1)
        self.assertIsNone(summarize(os.path.join(ROOT, "no-such-dir")))

    def test_snapshot_fallback_and_cap_counts(self):
        from display.decrypto import cap_stats, load
        with tempfile.TemporaryDirectory() as d:
            ep = os.path.join(d, "res", "model_seed0", "qwen3_4b_qwen3_4b_marshal_4b_000_000", "env_seed0", "episode_0")
            os.makedirs(ep)
            for role, n in (("encoder", 1022), ("decoder", 300), ("interceptor", 1022)):
                with open(os.path.join(ep, f"{role}_history.json"), "w") as f:
                    json.dump([{"role": "assistant", "content": "<think>\n" + "w " * n + "</think>\nANSWER: {}"}], f)
            cap = cap_stats(os.path.join(d, "res"), lambda t: len(t.split()))
            self.assertEqual(cap["qwen3_4b"]["encoder"]["capped"], 1)
            self.assertEqual(cap["qwen3_4b"]["decoder"]["capped"], 0)
            self.assertEqual(cap["marshal_4b"]["interceptor"]["cap_rate"], 1.0)
            snap = os.path.join(d, "snap.json")
            with open(snap, "w") as f:
                json.dump({"experiment": "x", "pairings": [{"episodes": 1}], "cap": cap}, f)
            self.assertEqual(load(os.path.join(d, "missing"), snap)["experiment"], "x")
            self.assertIsNone(load(os.path.join(d, "missing"), os.path.join(d, "none.json")))

    def test_clue_stems_and_rounds(self):
        from display.decrypto import clue_stats, episode_rounds, shares_stem
        self.assertTrue(shares_stem("editor", "edit"))
        self.assertTrue(shares_stem("generation", "generate"))
        self.assertFalse(shares_stem("problem", "issue"))
        self.assertFalse(shares_stem("card", "care"))
        history = [
            {"role": "user", "content": "[ENCODER] Turn 1.\nThe four keywords are:\n    {1: edit,\n    2: task,\n"
                                        "    3: issue,\n    4: device}\n\nThe code is 1-3-4, which ..."},
            {"role": "assistant", "content": "[ENCODER] ..."},
            {"role": "user", "content": "[DECODER] The hints given by the Encoder for this turn are:\n"
                                        "    {a: editor,\n    b: problem,\n    c: tool}\n"},
            {"role": "assistant", "content": '[DECODER] <think>x</think>\nANSWER: {"guess": "1-3-4"}'},
            {"role": "assistant", "content": '[INTERCEPTOR] maybe {"guess": "2-3-4"} no\nANSWER: {"guess": "1-3-4"}'},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "combined_history.json")
            with open(path, "w") as f:
                json.dump(history, f)
            rounds = episode_rounds(path)
        self.assertEqual(rounds[0]["hints"], ["editor", "problem", "tool"])
        self.assertEqual(rounds[0]["interceptor"], "1-3-4")
        stats = clue_stats(rounds)
        self.assertEqual((stats["clues"], stats["stem_clues"], stats["intercepts_with_stem"]), (3, 1, 1))
        self.assertEqual(stats["examples"], ["edit → editor"])


# ------------------------------------------------------------- server & export
class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from display import server
        cwd = os.getcwd()
        os.chdir(ROOT)
        cls.addClassCleanup(os.chdir, cwd)
        server.Handler.games = GameIndex(server.DEFAULT_GAMES)
        server.Handler.games.scan()
        server.Handler.runs = RunIndex(ROOT)
        server.Handler.live = None
        server.Handler.live_problem = "off"
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.addClassCleanup(cls.httpd.server_close)
        cls.addClassCleanup(cls.httpd.shutdown)

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return r.status, r.read()

    def test_page_and_static(self):
        status, body = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"MARSHAL", body)
        self.assertEqual(self.get("/static/app.js")[0], 200)

    def test_no_path_escape(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/static/../server.py")
        self.assertEqual(cm.exception.code, 404)

    def test_api(self):
        games = json.loads(self.get("/api/games")[1])["games"]
        self.assertTrue(games)
        rec = json.loads(self.get(f"/api/game?id={games[0]['id']}")[1])
        self.assertEqual(rec["id"], games[0]["id"])
        res = json.loads(self.get("/api/results")[1])
        self.assertIn("arena", res)
        self.assertEqual(len(res["runs"]), 7)
        self.assertFalse(json.loads(self.get("/api/live/options")[1])["enabled"])


class ExportTest(unittest.TestCase):
    def test_single_file(self):
        from display.export import build
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            html, n = build(limit=3)
        finally:
            os.chdir(cwd)
        self.assertLessEqual(n, 3)
        self.assertIn("window.MARSHAL_EMBED", html)
        self.assertNotIn('href="/static/app.css"', html)
        self.assertNotIn('src="/static/', html)
        self.assertIn("data:font/woff2;base64,", html)
        embed = html.split("window.MARSHAL_EMBED = ", 1)[1].split(";</script>", 1)[0]
        self.assertEqual(len(json.loads(embed)["games"]), n)


# --------------------------------------------------------------- model client
class FakeVllm(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "test/model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            body = json.dumps({"count": len(req["prompt"].split())}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        FakeVllm.last = req
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for piece in ["ok\n</think>", "<answer><BET></answer>"]:
            self.wfile.write(f"data: {json.dumps({'choices': [{'text': piece}]})}\n\n".encode())
        self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': 7, 'prompt_tokens': 3}})}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")


class PlayersTest(unittest.TestCase):
    def test_model_player_streams_completions(self):
        from display.players import ModelPlayer, render_prompt
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllm)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        p = ModelPlayer("T", f"http://127.0.0.1:{httpd.server_address[1]}/v1", "test/model")
        self.assertIsNone(p.check())
        self.assertEqual(p.count_tokens("a b c"), 3)
        seen = []
        out = p.generate("PROMPT", 100, on_partial=seen.append)
        self.assertEqual(out["text"], "ok\n</think><answer><BET></answer>")
        self.assertEqual(out["completion_tokens"], 7)
        self.assertEqual(seen[-1], out["text"])
        self.assertEqual(FakeVllm.last["top_k"], 100)
        self.assertIn("isn't answering", ModelPlayer("T", "http://127.0.0.1:9/v1", "m").check())
        prompt = render_prompt([{"role": "system", "content": "s"}, {"role": "user", "content": "u1"},
                                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"}])
        self.assertIn('<turn from="you">\na1\n</turn>', prompt)
        self.assertTrue(prompt.endswith("u2"))


@unittest.skipUnless(HAVE_SPIEL, "needs open_spiel (use the venv)")
class LiveTest(unittest.TestCase):
    def test_person_against_the_bot(self):
        from display.live import LiveManager
        with tempfile.TemporaryDirectory() as d:
            live = LiveManager(save_dir=d)
            self.assertIsNone(live.start({"game": "tictactoe", "seats": ["human", "bot"], "seed": 1}))
            for _ in range(10):
                for _ in range(200):
                    if live.awaiting or live.status != "running":
                        break
                    threading.Event().wait(0.02)
                if live.status != "running":
                    break
                legal = live.awaiting["legal"]
                self.assertIn("isn't a legal move", live.submit(0, "<X(9,9)>"))
                self.assertIsNone(live.submit(0, legal[0]))
                threading.Event().wait(0.05)
            live.thread.join(timeout=10)
            self.assertEqual(live.status, "finished")
            self.assertTrue(os.listdir(d))
            self.assertIn("Only one seat", live.start({"game": "tictactoe", "seats": ["bot", "bot"]}))
            self.assertIn("no built-in bot", live.start({"game": "hanabi", "seats": ["human", "bot"]}))


if __name__ == "__main__":
    unittest.main()
