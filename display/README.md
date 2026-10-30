# MARSHAL display

A full-screen display of MARSHAL for a monitor, built like the Decrypto one
(`../decrypto/display`). It alternates two screens:

- **Arena**: MARSHAL games, move by move. Each model's reasoning types out
  beside the board, then its move lands. Under each player are the per-turn
  rewards MARSHAL's trainer would assign. Visitors can take a seat.
- **Results**: what MARSHAL is, the Arena evaluation (base Qwen3-4B vs the
  published MARSHAL checkpoint), MARSHAL in Decrypto, the Midway and Polaris
  bring-up runs, their training curves, what the numbers say, and how the
  bring-up compares with the paper.

Every game runs **MARSHAL's own code**: the game classes in `roll/agentic/env/`,
their built-in bots, and a port of how `roll/agentic/rollout/env_manager.py`
builds each player's prompt, checks its answer and assigns turn rewards
(`display/engine.py`). A model that writes past its token limit or breaks the
`<think>…</think><answer>…</answer>` format forfeits (−11), as in training.
Check: with the Qwen3-4B tokenizer, the engine's Tic-Tac-Toe prompts are 506
tokens for X's first move and 628 for O's, the same `tokens/prompt_length` ROLL
logged in the Midway scale-up run.

## Replays and results (no installs)

```shell
python3 -m display.server            # run from the repo root
# open http://localhost:8766 on the monitor, press F for full screen
```

Only the standard library is needed. It plays the games in `display/data/games/`
(committed) and `results/display_games/` (yours), and builds Results from
`display/data/runs.json`, the Midway logs, and any run folders under `results/`.

### Full curves for the Polaris runs

Only milestone values from the Polaris notes are on this machine. Copy a run's
output folder into `results/` and its full TensorBoard curves replace the
milestones on the next page load (a TensorBoard reader is built in):

```shell
rsync -a polaris:/lus/eagle/.../MARSHAL/results/tictactoe_selfplay_polaris_megatron_long/7198659_* \
      results/tictactoe_selfplay_polaris_megatron_long/
```

The same works for any run: a folder with `tensorboard/` or `logs/custom_logs.log`
anywhere under `results/` appears as a "found" run.

### One file, no server

```shell
python3 -m display.export --out marshal_display.html          # all games (~2–4 MB)
python3 -m display.export --match connect --limit 10 --out c4.html
```

The page, fonts, images, games and Results data in one HTML file, for example on
a USB stick for the venue's computer.

## Live games and the Arena evaluation

Live games and recording need OpenSpiel. The system Python won't take `pip
install`, so use a venv in the repo (`.venv` is git-ignored):

```shell
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python open_spiel numpy matplotlib pillow imageio omegaconf pyyaml
```

### Model servers (AMD Strix Halo)

The players are listed in `display/strix_halo.yaml`. The two models run as vLLM
containers, started with the Decrypto repo's script (one GPU, both fit):

```shell
# base Qwen3-4B on :8000 (the Decrypto display's server; skip if it is already up)
../decrypto/scripts/strix_halo/serve_local_model.sh
# MARSHAL Generalist 4B on :8001, in its own container
CONTAINER_NAME=marshal-vllm GPU_MEM_UTIL=0.35 \
  ../decrypto/scripts/strix_halo/serve_local_model.sh nics-efc/MARSHAL-Generalist-Qwen3-4B 8001
# stop it again
docker rm -f marshal-vllm
```

`CONTAINER_NAME` matters: the script replaces any container with the same name.

### Live games

```shell
.venv/bin/python -m display.server --live
```

Press **L** (or click **Live game**), pick a game and who sits in each seat:
Qwen3-4B, MARSHAL 4B, Claude (headless `claude -p` on this machine's login, as in
the Decrypto display), a person, or the game's built-in bot (MCTS, CFR or Nash
equilibrium, at most one seat). Every screen open on the display follows the game.
A person moves with the buttons, or by clicking the board in Tic-Tac-Toe and
Connect Four. When a person plays poker or Hanabi, the screen hides what their seat
couldn't see (the other card, the other player's reasoning, their own Hanabi
hand) until the hand ends. Finished games are saved to `results/display_games/live/`
and join the replays. Models answer under the paper's limits (4096 tokens per
answer, 32768 per game), so one answer can take a minute or two on the iGPU.

### Recording games and the Arena evaluation

```shell
bash display/record_eval.sh        # both models vs MARSHAL's evaluation bots, both seats, + Hanabi
.venv/bin/python -m display.record --game connect_four --seats marshal-4b bot --games 10 --swap --parallel 4
.venv/bin/python -m display.record --game tictactoe --seats qwen3-4b qwen3-4b --games 4 \
    --max-new-tokens 1024 --sequence-length 4096 --tag bringup-settings   # the bring-up's limits
```

Games tagged `eval` feed the **Arena results** page; `bringup-settings` games
replay the Midway/Polaris limits, where O runs out of tokens on its first move.

## MARSHAL in Decrypto

`display/decrypto/marshal_vs_qwen3_4b.yaml` runs Decrypto's competitive setup
(team = Alice + Bob from one model, Eve from either) with base Qwen3-4B and
MARSHAL 4B, under Decrypto's own settings for both. Run it from the Decrypto repo:

```shell
cd ../decrypto
.venv/bin/python run.py --config-path=$PWD/../MARSHAL/display/decrypto --config-name=marshal_vs_qwen3_4b
```

Results land in `../decrypto/results/marshal_vs_qwen3_4b/`. The Decrypto display
replays them, and this display's **MARSHAL in Decrypto** page counts them per
pairing (`--decrypto DIR` points it elsewhere; `--decrypto ''` hides the page):
outcomes, answer-format retries per role, how often each model's clues copy
their keyword's stem (edit → editor) and how often Eve intercepts those rounds.

How often each model's thinking ran into Decrypto's 1024-token budget needs the
tokenizer, so it is computed once into a committed snapshot, which the page also
falls back to when the Decrypto repo isn't next to this one:

```shell
.venv/bin/python -m display.decrypto snapshot     # writes display/data/decrypto_summary.json (uses :8000/tokenize)
```

## What the numbers say

The **What the numbers say** page explains these; they come from the per-step
metrics in `display/data/metrics/` and MARSHAL's reward code:

- `critic/score/mean` is the **raw** game score (a player's summed turn rewards:
  +0.05 per valid answer, ±1 win/loss, +0.1 draw, −11 forfeit), not a normalized one.
- In every Midway run, **no game got past move 2**. The 0.5B model broke format on
  move 1. With Qwen3-4B, X opened validly (~220 tokens), then O's first answer hit
  the 1024-token cap (1027 tokens) before `</answer>`, so O forfeited every game.
  That is the −5.475 score, (0.05 − 11) / 2.
- With per-seat mean normalization, identical outcomes give zero advantage:
  `actor/pg_loss` is exactly 0 in 14 of the 20 scale-up steps. The bring-up proved
  the plumbing but carried essentially no learning signal.
- The 400-step Polaris run's score did move (−5.4 → −1.26). Its full curves are on
  Polaris (see above).

## Controls

| Key | Action |
| --- | --- |
| Space | Pause or resume |
| ← → | Step through the game, or page through Results |
| N | Next game |
| R / A | Results / Arena |
| G | Choose a saved game |
| L | Start a live game (`--live`) |
| P | Show or hide what the player is told |
| + − | Faster or slower playback |
| H | Hide the footer |
| F | Full screen |

Address options: `?speed=1.5`, `?scene=results`, `?slide=arena` (or `decrypto`,
`runs`, `curves`, `findings`, `paper`), `?game=<id>`, `?filter=connect marshal`,
`?tour=0` (stay on one screen), `?shuffle=0`, `?pause=1`.

On the monitor: `chromium --kiosk "http://localhost:8766/?speed=1.2"`, and turn
off the OS screen blanking. `--host 0.0.0.0` lets other machines open it (and
start live games), so use it only on a trusted network. The display uses port
8766 so it can run next to the Decrypto display on 8765.

## Files

| File | What it does |
| --- | --- |
| `engine.py` | One game with MARSHAL's env classes: prompts, answer check, forfeits, turn rewards, board views |
| `players.py` | Seats: vLLM model (raw `/v1/completions`, streamed), Claude (`claude -p`), person, built-in bot |
| `config.py`, `strix_halo.yaml` | The players on offer and the generation limits |
| `record.py`, `record_eval.sh` | Play and save games without the screen; the Arena evaluation set |
| `live.py` | The live game behind the server, with people's moves |
| `games.py` | Find saved games; count Arena results |
| `runs.py`, `tfevents.py` | Training metrics from SLURM/ROLL logs and TensorBoard files |
| `decrypto.py`, `decrypto/` | MARSHAL in Decrypto: the experiment config and its summary |
| `data/decrypto_summary.json` | Snapshot of that summary with the thinking-budget counts |
| `data/runs.json` | The run ledger, milestones, paper numbers and findings, with sources |
| `data/metrics/` | Per-step metrics extracted from the three Midway logs |
| `data/games/` | The recorded games (`*.marshal-game.json`) |
| `server.py`, `export.py` | Web server (replays, results, live) and the single-file export |
| `static/` | The page (HTML, CSS, JS), mascot, method figure and fonts |
| `tests/` | `python3 -m unittest discover -s display/tests -t .` (the venv runs all of them) |

The fonts (Atkinson Hyperlegible Next and Mono) are from Google Fonts under the
SIL Open Font License, bundled so the display works offline. The mascot and the
method figure are from `assets/MARSHAL_assets/`.
