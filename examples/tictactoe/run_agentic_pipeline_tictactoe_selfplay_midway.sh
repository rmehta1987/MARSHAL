#!/bin/bash
# Midway-flavored launcher for the MARSHAL tictactoe self-play smoke run.
# Invoked from scripts/train_midway.sbatch (or directly, after activating
# the marshal-train mamba env and setting ROLL_OUTPUT_DIR).

set +x
ulimit -u $(ulimit -Hu) 2>/dev/null || true

# ── Aggressive Ray cleanup ──
# Stale ray sessions from prior jobs poison the new head node. Same preamble
# as the original tictactoe launcher.
ray stop --force 2>/dev/null
sleep 2
pkill -u "$(whoami)" -f "ray::" 2>/dev/null
pkill -u "$(whoami)" -f "raylet" 2>/dev/null
pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null
pkill -u "$(whoami)" -f "runtime_env_agent" 2>/dev/null
sleep 3
rm -rf /tmp/ray/session_latest 2>/dev/null

REMAINING_RAY=$(pgrep -u "$(whoami)" -f "ray[: ]" -c 2>/dev/null) || REMAINING_RAY=0
if [ "$REMAINING_RAY" -gt 0 ]; then
    echo "WARNING: $REMAINING_RAY Ray processes still alive after cleanup, force killing..."
    pkill -9 -u "$(whoami)" -f "ray" 2>/dev/null
    sleep 2
fi
echo "=== Ray cleanup complete, remaining ray processes: $(pgrep -u $(whoami) -f 'ray[: ]' -c 2>/dev/null || echo 0) ==="

# Hydra wants config_path resolved relative to examples/; the original
# launcher does this via $(basename $(dirname $0)).
CONFIG_PATH=$(basename $(dirname $0))
ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# ROLL_OUTPUT_DIR may be set by the sbatch wrap; default if running interactively.
if [ -z "$ROLL_OUTPUT_DIR" ]; then
    ROLL_OUTPUT_DIR="/project/rcc/mehta5/MARSHAL/results/tictactoe_selfplay_midway_smoke/$(date +%Y%m%d-%H%M%S)"
fi
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR ROLL_LOG_DIR ROLL_RENDER_DIR
mkdir -p "$ROLL_LOG_DIR" "$ROLL_RENDER_DIR"

echo "CONFIG_PATH: $CONFIG_PATH"
echo "ROLL_PATH:   $ROLL_PATH"
echo "ROLL_OUTPUT_DIR: $ROLL_OUTPUT_DIR"

# ── Triton / libcuda symbol-resolution fix ──
# Triton 3.2.0 builds cuda_utils.so with NO NEEDED libcuda.so.1: the driver
# symbols (cuModuleGetFunction, cuModuleLoadData, ...) are left UND, expected
# to resolve from an already globally-loaded driver. deepspeed's triton ops
# trigger that dlopen during `import transformers` — before torch has loaded
# libcuda into the global namespace — so the import dies with
#   ImportError: .../cuda_utils.so: undefined symbol: cuModuleGetFunction
# Preload the host driver libcuda that `apptainer --nv` stages into
# /.singularity.d/libs so its symbols are in the global namespace from the start
# (this is the correct 535 host driver — NOT the container's baked 470 stub or
# the 550 compat lib the image fails to rm; those rm errors are harmless).
HOST_LIBCUDA=$(ls /.singularity.d/libs/libcuda.so.1 2>/dev/null || ls /.singularity.d/libs/libcuda.so* 2>/dev/null | head -1)
if [ -n "$HOST_LIBCUDA" ]; then
    export LD_PRELOAD="${HOST_LIBCUDA}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "LD_PRELOAD=$LD_PRELOAD (host driver libcuda for triton symbol resolution)"
else
    echo "WARNING: no host libcuda in /.singularity.d/libs — is apptainer --nv active?"
fi

python examples/start_agentic_pipeline.py \
  --config_path $CONFIG_PATH \
  --config_name agentic_val_tictactoe_selfplay_midway_smoke \
  2>&1 | tee "$ROLL_LOG_DIR/custom_logs.log"
