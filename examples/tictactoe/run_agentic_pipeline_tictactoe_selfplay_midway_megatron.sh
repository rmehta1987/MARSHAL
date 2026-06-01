#!/bin/bash
# Midway MEGATRON launcher (Qwen3-4B, TP=4, 3 steps) — identical to
# run_agentic_pipeline_tictactoe_selfplay_midway.sh except for the config_name
# and the default output-dir label. Invoked from scripts/train_midway_megatron.sbatch.

set +x
# Propagate python's exit code through the `python ... | tee` pipe at the end —
# without this the launcher returns tee's status (always 0) and sacct reports a
# crashed run as COMPLETED. (The deepspeed launchers have this latent masking;
# it didn't bite because those runs actually succeeded.)
set -o pipefail
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
    ROLL_OUTPUT_DIR="/project/rcc/mehta5/MARSHAL/results/tictactoe_selfplay_midway_megatron/$(date +%Y%m%d-%H%M%S)"
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

# ── Megatron tensor-parallel requirement ──
# With tensor_model_parallel_size>1 + sequence_parallel, Megatron requires a
# single CUDA work queue so the TP all-reduce/all-gather overlap is ordered
# correctly. ROLL/mcore_adapter don't set this; the upstream launcher relied on
# the CMU container env. Set it explicitly for the megatron path.
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS (megatron TP+SP ordering)"

# ── TransformerEngine attention backend ──
# TE's default cuDNN fused-attention backend fails on this image with
#   cuDNN Error: [cudnn_frontend] No execution plans support the graph
# (fused_attn_f16_arbitrary_seqlen.cu) for Qwen3's attn graph (GQA 32/8 +
# sequence_parallel) — the container's cuDNN has no plan for it on sm_90 over
# the 535 / CUDA-12.2 host driver (MVC). Route TE attention through flash-attn
# (2.7.2 is in the image; it uses its own kernels, not cuDNN) instead.
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
echo "NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN NVTE_FLASH_ATTN=$NVTE_FLASH_ATTN (TE -> flash-attn, avoid cuDNN fused attn)"

python examples/start_agentic_pipeline.py \
  --config_path $CONFIG_PATH \
  --config_name agentic_val_tictactoe_selfplay_midway_megatron \
  2>&1 | tee "$ROLL_LOG_DIR/custom_logs.log"
