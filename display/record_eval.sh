#!/usr/bin/env bash
# Record the display's evaluation set: each model against MARSHAL's evaluation
# opponents, moving first in half the games, plus Mini Hanabi self-play.
# These are the games the display replays and the "Arena" results slide counts.
#
#   bash display/record_eval.sh                 # both models, default sizes
#   MODELS="marshal-4b" N_GRID=4 bash display/record_eval.sh
#
# Needs both model servers from display/strix_halo.yaml running, and the venv
# (.venv) with open_spiel. Each model's games run concurrently; vLLM batches them.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-.venv/bin/python}"
OUT="${OUT:-display/data/games}"
LOGS="${LOGS:-results/display_games/logs}"
MODELS="${MODELS:-qwen3-4b marshal-4b}"
N_GRID="${N_GRID:-8}"      # Tic-Tac-Toe and Connect Four vs MCTS (100 simulations)
N_POKER="${N_POKER:-12}"   # Kuhn vs CFR and Leduc vs Nash equilibrium (12 = every Kuhn deal, both seats)
N_HANABI="${N_HANABI:-4}"  # Mini Hanabi self-play
mkdir -p "$OUT" "$LOGS"

run() {  # run <log-name> <args...>
    local name="$1"; shift
    "$PY" -m display.record --out "$OUT" --tag eval "$@" >"$LOGS/$name.log" 2>&1 &
}

for m in $MODELS; do
    run "$m-tictactoe" --game tictactoe --seats "$m" bot --games "$N_GRID" --swap --parallel 4
    run "$m-connect_four" --game connect_four --seats "$m" bot --games "$N_GRID" --swap --parallel 4
    run "$m-kuhn" --game kuhn_poker --seats "$m" bot --games "$N_POKER" --swap --parallel 3
    run "$m-leduc" --game leduc_poker --seats "$m" bot --games "$N_POKER" --swap --parallel 3
    run "$m-hanabi" --game hanabi --seats "$m" "$m" --games "$N_HANABI" --parallel 2
done
wait
echo "Done. Games in $OUT, logs in $LOGS."
