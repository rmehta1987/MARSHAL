#!/bin/bash
# Polaris MEGATRON launcher (Qwen3-4B, megatron_train TP=4, 3 steps) — merge of
# the GREEN Polaris smoke launcher (run_agentic_pipeline_tictactoe_selfplay_polaris.sh,
# jid 7186746: Ray cleanup, thread caps, RAY_NUM_CPUS, VLLM_USE_V1) and the
# Midway megatron launcher's additions (run_..._midway_megatron.sh, GREEN jid
# 50261211: CUDA_DEVICE_MAX_CONNECTIONS=1, NVTE_FUSED_ATTN=0/NVTE_FLASH_ATTN=1).
# Invoked by scripts/train_polaris_megatron.pbs.

set -o pipefail   # final `python ... | tee` must report python's real exit code,
                  # not tee's 0 (else a crash reads as success). See midway_notes.

ulimit -u $(ulimit -Hu) 2>/dev/null || true

# ── Aggressive Ray cleanup ──
# Stale ray sessions from a prior job poison the new head node. Same preamble as
# the upstream tictactoe launcher.
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

# Hydra resolves config_path relative to examples/; the upstream launcher derives
# it from $(basename $(dirname $0)) -> 'tictactoe'.
CONFIG_PATH=$(basename $(dirname $0))
ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# ROLL_OUTPUT_DIR may be set by the PBS wrap; default if running interactively.
if [ -z "$ROLL_OUTPUT_DIR" ]; then
    ROLL_OUTPUT_DIR="$ROLL_PATH/results/tictactoe_selfplay_polaris_megatron/$(date +%Y%m%d-%H%M%S)"
fi
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR ROLL_LOG_DIR ROLL_RENDER_DIR
mkdir -p "$ROLL_LOG_DIR" "$ROLL_RENDER_DIR"

echo "CONFIG_PATH: $CONFIG_PATH"
echo "ROLL_PATH:   $ROLL_PATH"
echo "ROLL_OUTPUT_DIR: $ROLL_OUTPUT_DIR"

# ── Defensive libcuda preload (triton 3.2.0 symbol-resolution insurance) ──
# Triton builds cuda_utils.so leaving the driver symbols UND, expecting an
# already globally-loaded libcuda. Preloading the host driver libcuda is
# harmless insurance against the import-order race (see the smoke launcher).
HOST_LIBCUDA=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
[ -z "$HOST_LIBCUDA" ] && HOST_LIBCUDA=$(ls /usr/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null | head -1)
if [ -n "$HOST_LIBCUDA" ]; then
    export LD_PRELOAD="${HOST_LIBCUDA}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "LD_PRELOAD=$LD_PRELOAD (host driver libcuda for triton symbol resolution)"
else
    echo "NOTE: no libcuda.so.1 found for preload (expected on a login node; should be present on a compute node)"
fi

# ── Cap per-process CPU thread pools (REQUIRED on Polaris) ──
# The PBS job cgroup hard-caps the whole job at pids.max=4096 threads+processes
# (measured, jid 7186710). Default OpenBLAS/OMP pools spawn one thread per
# hardware thread (64) per process; with 6 GPU workers (4 megatron TP ranks +
# 1 vLLM + 1 reference) plus env/scheduler actors that blows the cap and
# workers die at construction with EAGAIN. Must be exported BEFORE python so
# every Ray worker inherits it before its first numpy/OpenBLAS import.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Cap Ray's CPU count so it prestarts ~16 idle workers instead of one per
# detected core (~64; each idle worker carries ~30+ Ray threads). ROLL's
# start_ray_cluster() reads RAY_NUM_CPUS and passes it to `ray start --num-cpus`.
export RAY_NUM_CPUS=16
# Cap ninja/compiler bursts for any JIT that fires under the crowded Ray job.
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
# Keep the vLLM V1 engine. V1 is markedly LIGHTER on startup threads than V0
# (V0 died at RolloutScheduler.__init__ under pids.max=4096; V1 reaches the
# loop — measured on the smoke, jids 7186716/7186717). The megatron->vllm
# weight-sync goes through the NCCL-broadcast bucket path, which is V1-safe
# (only bucket metadata is msgpack'd; the tensor moves over NCCL).
export VLLM_USE_V1=1

# ── Megatron tensor-parallel requirement ──
# With tensor_model_parallel_size>1 + sequence_parallel, Megatron requires a
# single CUDA work queue so the TP all-reduce/all-gather overlap is ordered
# correctly. ROLL/mcore_adapter don't set it; the upstream CMU launcher relied
# on its container env. Proven necessary on Midway (GREEN jid 50261211).
export CUDA_DEVICE_MAX_CONNECTIONS=1
echo "CUDA_DEVICE_MAX_CONNECTIONS=$CUDA_DEVICE_MAX_CONNECTIONS (megatron TP+SP ordering)"

# ── TransformerEngine attention backend ──
# On Midway, TE's default cuDNN fused-attention backend had no execution plan
# for Qwen3's attn graph (GQA 32/8 + sequence_parallel) and crashed the first
# actor_train forward; routing TE through flash-attn's own kernels fixed it
# (GREEN jid 50261211). Polaris has a different cuDNN, so fused attn *might*
# work here — but the flags are proven and cheap; keep them.
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
echo "NVTE_FUSED_ATTN=$NVTE_FUSED_ATTN NVTE_FLASH_ATTN=$NVTE_FLASH_ATTN (TE -> flash-attn, avoid cuDNN fused attn)"

python examples/start_agentic_pipeline.py \
  --config_path $CONFIG_PATH \
  --config_name ${MARSHAL_CONFIG_NAME:-agentic_val_tictactoe_selfplay_polaris_megatron} \
  2>&1 | tee "$ROLL_LOG_DIR/custom_logs.log"
