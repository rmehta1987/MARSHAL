"""Who can sit at a MARSHAL table on the display.

- ModelPlayer: a model behind an OpenAI-compatible vLLM server. It gets the exact
  training prompt text (see engine.chat_text) through /v1/completions, so base
  Qwen3-4B and a MARSHAL checkpoint are asked the same way ROLL asked them.
- ClaudePlayer: Claude through headless Claude Code (`claude -p`) on this
  machine's login, with no tools, settings or MCP servers, in an empty folder.
- HumanPlayer: a person choosing a legal move on the screen.
- BotPlayer: the game's own built-in opponent (MCTS, CFR, Nash equilibrium).

Only the standard library is used here.
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request


class PlayerError(RuntimeError):
    pass


class BotPlayer:
    kind = "bot"

    def __init__(self, name="Built-in bot"):
        self.name = name

    def describe(self):
        return {"kind": self.kind, "name": self.name}


class HumanPlayer:
    kind = "human"

    def __init__(self, name="Human"):
        self.name = name

    def describe(self):
        return {"kind": self.kind, "name": self.name}


class ModelPlayer:
    kind = "model"

    def __init__(self, name, url, model, temperature=0.6, top_p=0.99, top_k=100, timeout=1200, note=None):
        self.name = name
        self.url = url.rstrip("/")  # e.g. http://localhost:8000/v1
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.note = note

    def describe(self):
        info = {"kind": self.kind, "name": self.name, "model": self.model,
                "sampling": {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k}}
        if self.note:
            info["note"] = self.note
        return info

    @property
    def root(self):
        return self.url[:-3] if self.url.endswith("/v1") else self.url

    def _post(self, url, payload, timeout):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            raise PlayerError(f"{self.name}: server said {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise PlayerError(f"{self.name} isn't answering at {self.url} ({exc}).") from exc

    def check(self):
        """None when the server answers and serves this model, else the problem."""
        try:
            with urllib.request.urlopen(self.url + "/models", timeout=5) as r:
                served = [m.get("id") for m in json.load(r).get("data", [])]
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            return f"{self.name} isn't answering at {self.url} ({exc}). Start its model server, then try again."
        if self.model not in served:
            return f"{self.url} serves {', '.join(served) or 'nothing'}, not {self.model}."
        return None

    def count_tokens(self, text):
        with self._post(self.root + "/tokenize", {"model": self.model, "prompt": text,
                                                  "add_special_tokens": False}, 60) as r:
            data = json.load(r)
        return int(data.get("count") or len(data.get("tokens") or []))

    def generate(self, prompt, max_tokens, on_partial=None, stop_event=None):
        """Stream one completion. Returns {"text", "completion_tokens", "prompt_tokens", "finish_reason"}."""
        payload = {
            "model": self.model, "prompt": prompt, "max_tokens": int(max_tokens),
            "temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
            "stream": True, "stream_options": {"include_usage": True},
            "skip_special_tokens": True,
        }
        text, usage, finish = [], None, None
        resp = self._post(self.url + "/completions", payload, self.timeout)
        try:
            for raw in resp:
                if stop_event is not None and stop_event.is_set():
                    break  # closing the connection makes vLLM abort the request
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("text"):
                        text.append(choice["text"])
                        if on_partial:
                            on_partial("".join(text))
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        finally:
            resp.close()
        return {
            "text": "".join(text),
            "completion_tokens": (usage or {}).get("completion_tokens") or 0,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "finish_reason": finish,
        }


# ------------------------------------------------------------------ Claude seat
# Set only when the display itself runs inside a Claude Code session. Passing them
# on would tie each move to that session instead of starting a fresh one.
PARENT_SESSION_VARS = (
    "CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS",
)


def render_prompt(messages):
    """Fold the game conversation (ending with the new user message) into one prompt."""
    turns = [m for m in messages if m["role"] != "system"]
    *earlier, last = turns
    if not earlier:
        return last["content"]
    parts = ["This game is already under way. Here is the conversation so far. "
             'Turns marked from="you" are your own earlier replies.', ""]
    for m in earlier:
        who = "you" if m["role"] == "assistant" else "game"
        parts.append(f'<turn from="{who}">\n{m["content"]}\n</turn>')
    parts += ["", "The game's new message, which you are answering now:", "", last["content"]]
    return "\n".join(parts)


class ClaudePlayer:
    kind = "claude"

    def __init__(self, name="Claude", model_id="claude-opus-5-5", effort=None, timeout=300,
                 claude_command="claude", show_thinking=True):
        self.name = name
        self.model_id = model_id
        self.effort = effort
        self.timeout = timeout
        self.claude_command = claude_command
        self.show_thinking = show_thinking
        self.workdir = tempfile.mkdtemp(prefix="marshal-claude-")

    def describe(self):
        return {"kind": self.kind, "name": self.name, "model": self.model_id, "effort": self.effort,
                "note": "Plays through `claude -p`. Answers are read from the last <answer>…</answer>; "
                        "the token limits that apply to trained models don't apply."}

    def env(self):
        return {k: v for k, v in os.environ.items() if k not in PARENT_SESSION_VARS}

    def check(self):
        exe = shutil.which(self.claude_command)
        if exe is None:
            return f"The Claude Code CLI ('{self.claude_command}') isn't installed or isn't on PATH."
        try:
            out = subprocess.run([exe, "auth", "status", "--json"], env=self.env(),
                                 capture_output=True, text=True, timeout=30)
            status = json.loads(out.stdout or "{}")
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            return f"Couldn't check the Claude Code login ({exc})."
        if not status.get("loggedIn"):
            return "Claude Code isn't signed in. Run `claude auth login` on this machine, then try again."
        return None

    def command(self, system_prompt):
        exe = shutil.which(self.claude_command) or self.claude_command
        cmd = [exe, "-p", "--model", self.model_id,
               "--output-format", "stream-json", "--verbose", "--include-partial-messages",
               "--tools", "", "--system-prompt", system_prompt, "--setting-sources", "",
               "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
        if self.show_thinking:
            cmd += ["--settings", json.dumps({"showThinkingSummaries": True})]
        if self.effort:
            cmd += ["--effort", str(self.effort)]
        return cmd

    def move(self, messages, on_partial=None, stop_event=None):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        try:
            proc = subprocess.Popen(self.command(system), cwd=self.workdir, env=self.env(),
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8")
        except FileNotFoundError as exc:
            raise PlayerError(f"Claude Code CLI '{self.claude_command}' not found.") from exc
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            proc.kill()

        timer = threading.Timer(self.timeout, kill)
        timer.start()
        errors = []
        reader = threading.Thread(target=lambda: errors.append(proc.stderr.read()), daemon=True)
        reader.start()
        watcher = None
        if stop_event is not None:
            watcher = threading.Thread(target=lambda: stop_event.wait() and proc.poll() is None and proc.kill(),
                                       daemon=True)
            watcher.start()
        text, thinking, final_thinking, result = [], [], [], None
        try:
            proc.stdin.write(render_prompt(messages))
            proc.stdin.close()
            for line in proc.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                kind = event.get("type")
                if kind == "stream_event":
                    ev = event.get("event") or {}
                    if ev.get("type") != "content_block_delta":
                        continue
                    delta = ev.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text.append(delta.get("text", ""))
                    elif delta.get("type") == "thinking_delta":
                        thinking.append(delta.get("thinking", ""))
                    else:
                        continue
                    if on_partial:
                        on_partial("".join(thinking), "".join(text))
                elif kind == "assistant":
                    for block in (event.get("message") or {}).get("content") or []:
                        if block.get("type") == "thinking" and block.get("thinking"):
                            final_thinking.append(block["thinking"])
                elif kind == "result":
                    result = event
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            reader.join(timeout=5)
            for pipe in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    pipe.close()
                except OSError:
                    pass
        if stop_event is not None and stop_event.is_set():
            return {"text": "", "thinking": ""}
        if timed_out.is_set():
            raise PlayerError(f"claude -p took longer than {self.timeout} s")
        if result is None or result.get("is_error") or result.get("subtype") != "success":
            detail = (result or {}).get("result") or "".join(errors).strip()[-500:]
            raise PlayerError(f"claude -p failed: {detail or f'exit code {proc.returncode}'}")
        answer = result.get("result") or "".join(text)
        return {"text": answer, "thinking": "".join(thinking) or "\n\n".join(final_thinking)}


def make_player(spec):
    """Build a player from a config entry (a dict with `kind`)."""
    kind = spec.get("kind", "model")
    if kind == "model":
        return ModelPlayer(spec["name"], spec["url"], spec["model"],
                           temperature=spec.get("temperature", 0.6), top_p=spec.get("top_p", 0.99),
                           top_k=spec.get("top_k", 100), timeout=spec.get("timeout", 1200),
                           note=spec.get("note"))
    if kind == "claude":
        return ClaudePlayer(spec.get("name", "Claude"), spec.get("model_id", "claude-opus-5-5"),
                            effort=spec.get("effort"), timeout=spec.get("timeout", 300),
                            claude_command=spec.get("claude_command", "claude"))
    if kind == "human":
        return HumanPlayer(spec.get("name", "Human"))
    if kind == "bot":
        return BotPlayer(spec.get("name", "Built-in bot"))
    raise ValueError(f"Unknown player kind {kind!r}")
