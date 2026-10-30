"""Load the display's player config (YAML or JSON) and build seats from it."""

import json
import os

from display.players import make_player

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strix_halo.yaml")


def load_config(path=None):
    path = path or DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if path.endswith(".json"):
        cfg = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # live mode runs in the venv, which has PyYAML
            raise SystemExit(f"Reading {path} needs PyYAML (pip install pyyaml), or use a .json config.") from exc
        cfg = yaml.safe_load(text)
    cfg.setdefault("players", [])
    cfg.setdefault("generation", {})
    cfg.setdefault("defaults", {})
    cfg["path"] = path
    return cfg


class Roster:
    """The players a config offers, by key."""

    def __init__(self, cfg):
        self.cfg = cfg
        gen = cfg.get("generation") or {}
        self.specs = {}
        for spec in cfg["players"]:
            spec = dict(spec)
            if spec.get("kind", "model") == "model":
                for k in ("temperature", "top_p", "top_k"):
                    spec.setdefault(k, gen.get(k, {"temperature": 0.6, "top_p": 0.99, "top_k": 100}[k]))
            self.specs[spec["key"]] = spec
        for key, kind, name in (("human", "human", "Human"), ("bot", "bot", "Built-in bot")):
            self.specs.setdefault(key, {"key": key, "kind": kind, "name": name})
        self.max_new_tokens = int(gen.get("max_new_tokens", 4096))
        self.sequence_length = int(gen.get("sequence_length", 32768))

    def build(self, key):
        if key not in self.specs:
            raise ValueError(f"No player '{key}' in {self.cfg.get('path')}. Choose from {', '.join(self.specs)}.")
        return make_player(self.specs[key])

    def options(self):
        return [{"key": k, "kind": s.get("kind", "model"), "name": s.get("name", k), "note": s.get("note")}
                for k, s in self.specs.items()]
