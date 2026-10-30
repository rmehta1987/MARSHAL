"""Training results for the display: the run ledger plus per-step metrics.

The ledger (display/data/runs.json) lists the GREEN bring-up runs from
midway_notes.md and the Polaris notes. Each run's per-step metrics come from
the first source found:

1. TensorBoard event files in the run's output folder (`results_glob`), once
   copied from the cluster, e.g. results/tictactoe_selfplay_polaris_megatron_long/7198659_*/;
2. ROLL's driver log in that folder (logs/custom_logs.log, logs/log_rank_DRIVER_0_1.log);
3. a metrics file extracted from the run's SLURM log (display/data/metrics/<id>.json);
4. the SLURM log itself (`log`), full per-step JSON or the trimmed "step N: k=v" lines.

Runs with none of these show the milestone values the notes record. Output
folders and SLURM logs that the ledger doesn't list are added as found runs.

Extract a SLURM log's metrics (keeps everything except memory/turn-by-turn keys):
    python -m display.runs extract logs/train_midway_50252477.out display/data/metrics/midway-smoke.json
"""

import glob
import json
import os
import re
import sys

from display.tfevents import read_dir

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "data", "runs.json")
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Short names in the trimmed scale-up log -> the metric keys they came from.
TRIMMED = {
    "samples": "system/samples", "tps": "system/tps", "pg_loss": "actor/pg_loss", "kl_loss": "actor/kl_loss",
    "total_loss": "actor/total_loss", "approxkl": "actor/approxkl", "grad_norm": "actor_train/grad_norm",
    "lr": "actor/lr", "score": "critic/score/mean", "reward": "critic/rewards/mean",
    "adv": "critic/advantages/mean", "returns": "critic/returns/mean", "resp_len": "tokens/response_length/mean",
    "train_step_s": "time/actor_train/train_step/total",
}

KEEP_TIME = {"time/actor_train/train_step/total", "time/step_total", "time/rollout", "time/step_generate"}


def keep_key(key):
    if "memory" in key:
        return False
    if re.search(r"_turn_\d+$", key):
        return False
    if key.startswith(("time/", "val/time/")) and key not in KEEP_TIME:
        return False
    return True


def parse_log(path):
    """{step: {key: value}} from a SLURM/ROLL log (JSON metric lines or trimmed lines)."""
    steps = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = ANSI.sub("", line)
            if '"system/step"' in line:
                start = line.find("{")
                try:
                    data = json.loads(line[start:line.rindex("}") + 1])
                except ValueError:
                    continue
                step = int(data.get("system/step", len(steps)))
                steps.setdefault(step, {}).update(
                    {k: v for k, v in data.items() if keep_key(k) and isinstance(v, (int, float))})
                continue
            m = re.match(r"\s*step\s+(\d+):\s+(.*)", line)
            if m and "=" in m.group(2):
                step = int(m.group(1))
                for pair in m.group(2).split():
                    k, _, v = pair.partition("=")
                    if k in TRIMMED:
                        try:
                            steps.setdefault(step, {})[TRIMMED[k]] = float(v)
                        except ValueError:
                            pass
    return steps


def from_tensorboard(directory):
    steps = {}
    for tag, points in read_dir(directory).items():
        if not keep_key(tag):
            continue
        for step, value in points.items():
            steps.setdefault(int(step), {})[tag] = value
    return steps


def to_series(steps):
    series = {}
    for step in sorted(steps):
        for key, value in steps[step].items():
            if value is None or value != value:
                continue
            series.setdefault(key, []).append([step, round(float(value), 6)])
    return series


def log_header(path):
    """Job facts printed at the top of the SLURM logs written by scripts/train_midway*.sbatch."""
    meta = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            head = [ANSI.sub("", next(f, "")) for _ in range(200)]
    except OSError:
        return meta
    for line in head:
        m = re.search(r"=== Job (\d+) on (\S+) at (.*) ===", line)
        if m:
            meta.update(jid=m.group(1), node=m.group(2), started=m.group(3).strip())
        m = re.search(r"CONFIG_NAME:\s*(\S+)", line)
        if m:
            meta["config"] = m.group(1)
    return meta


class RunIndex:
    """The ledger runs plus any extra runs found on disk."""

    def __init__(self, repo_root=".", ledger_path=LEDGER, extra_roots=()):
        self.repo_root = os.path.abspath(repo_root)
        self.ledger_path = ledger_path
        self.extra_roots = list(extra_roots)
        self.ledger = {}
        self.runs = []
        self._series = {}
        self.scan()

    def _path(self, rel):
        return rel if os.path.isabs(rel) else os.path.join(self.repo_root, rel)

    def _load_series(self, run):
        """Fill run['source'] and return its series, or {} when only milestones exist."""
        for pattern in ([run["results_glob"]] if run.get("results_glob") else []):
            for folder in sorted(glob.glob(self._path(pattern))):
                tb = os.path.join(folder, "tensorboard")
                if os.path.isdir(tb) and glob.glob(os.path.join(tb, "**", "events.out.tfevents.*"), recursive=True):
                    steps = from_tensorboard(tb)
                    if steps:
                        run["source"] = os.path.relpath(tb, self.repo_root)
                        return to_series(steps)
                for name in ("custom_logs.log", "log_rank_DRIVER_0_1.log"):
                    log = os.path.join(folder, "logs", name)
                    if os.path.isfile(log):
                        steps = parse_log(log)
                        if steps:
                            run["source"] = os.path.relpath(log, self.repo_root)
                            return to_series(steps)
        if run.get("metrics") and os.path.isfile(self._path(run["metrics"])):
            with open(self._path(run["metrics"]), encoding="utf-8") as f:
                data = json.load(f)
            run["source"] = data.get("extracted_from") or run["metrics"]
            return data.get("series", {})
        if run.get("log") and os.path.isfile(self._path(run["log"])):
            steps = parse_log(self._path(run["log"]))
            if steps:
                run["source"] = run["log"]
                return to_series(steps)
        run["source"] = "milestones from the notes" if run.get("milestones") else None
        return {}

    def scan(self):
        with open(self.ledger_path, encoding="utf-8") as f:
            self.ledger = json.load(f)
        runs, series, claimed = [], {}, set()
        for entry in self.ledger.get("runs", []):
            run = dict(entry)
            s = self._load_series(run)
            series[run["id"]] = s
            run["points"] = max((len(v) for v in s.values()), default=0)
            if run.get("log"):
                claimed.add(os.path.abspath(self._path(run["log"])))
            if run.get("results_glob"):
                claimed.update(os.path.abspath(p) for p in glob.glob(self._path(run["results_glob"])))
            runs.append(run)
        runs += self._discover(claimed, series)
        self.runs, self._series = runs, series
        return runs

    def _discover(self, claimed, series):
        found = []
        roots = [os.path.join(self.repo_root, "results")] + [self._path(r) for r in self.extra_roots]
        folders = set()
        for root in roots:
            for tb in glob.glob(os.path.join(root, "**", "tensorboard"), recursive=True):
                folders.add(os.path.dirname(tb))
            for log in glob.glob(os.path.join(root, "**", "logs", "custom_logs.log"), recursive=True):
                folders.add(os.path.dirname(os.path.dirname(log)))
        for folder in sorted(folders):
            if os.path.abspath(folder) in claimed:
                continue
            rel = os.path.relpath(folder, self.repo_root)
            run = {"id": "found-" + re.sub(r"[^A-Za-z0-9_-]+", "-", rel).strip("-"), "title": rel,
                   "cluster": "Found on disk", "status": "found", "results_glob": rel}
            s = self._load_series(run)
            if s:
                run["points"] = max(len(v) for v in s.values())
                run["steps"] = run["points"]
                series[run["id"]] = s
                found.append(run)
        for log in sorted(glob.glob(os.path.join(self.repo_root, "logs", "*.out"))):
            if os.path.abspath(log) in claimed:
                continue
            steps = parse_log(log)
            if not steps:
                continue
            rel = os.path.relpath(log, self.repo_root)
            meta = log_header(log)
            run = {"id": "found-" + os.path.basename(log).replace(".", "-"), "title": rel, "cluster": "Found on disk",
                   "status": "found", "log": rel, "source": rel, "jid": meta.get("jid"), "steps": len(steps)}
            series[run["id"]] = to_series(steps)
            run["points"] = len(steps)
            found.append(run)
        return found

    def list(self):
        return [{k: v for k, v in r.items() if k not in ("milestones",)} for r in self.runs]

    def get(self, run_id):
        for run in self.runs:
            if run["id"] == run_id:
                return {**run, "series": self._series.get(run_id, {})}
        return None

    def overview(self):
        """Everything the results screens show, in one payload."""
        meta = {k: v for k, v in self.ledger.items() if k != "runs"}
        return {**meta, "runs": [self.get(r["id"]) for r in self.runs]}


def extract(log_path, out_path, note=None):
    steps = parse_log(log_path)
    if not steps:
        raise SystemExit(f"No per-step metrics in {log_path}.")
    data = {"extracted_from": note or log_path, "steps": len(steps), "series": to_series(steps)}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"{out_path}: {len(steps)} steps, {len(data['series'])} metrics")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) >= 3 and argv[0] == "extract":
        extract(argv[1], argv[2], note=argv[3] if len(argv) > 3 else None)
        return 0
    index = RunIndex(".")
    for run in index.runs:
        print(f"{run['id']:<40} {run.get('points', 0):>4} points  {run.get('source')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
