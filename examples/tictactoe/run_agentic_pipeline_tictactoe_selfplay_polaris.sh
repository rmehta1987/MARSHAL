#!/bin/bash
# Polaris-flavored launcher for the MARSHAL tictactoe self-play smoke.
# NATIVE venv path (NO apptainer/container) — analog of
# run_agentic_pipeline_tictactoe_selfplay_midway.sh minus the container glue.
# Invoked by scripts/train_polaris.pbs (which sets up modules, the venv, caches,
# and ROLL_OUTPUT_DIR), or directly after activating the marshal-train venv.

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
    ROLL_OUTPUT_DIR="$ROLL_PATH/results/tictactoe_selfplay_polaris_smoke/$(date +%Y%m%d-%H%M%S)"
fi
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR ROLL_LOG_DIR ROLL_RENDER_DIR
mkdir -p "$ROLL_LOG_DIR" "$ROLL_RENDER_DIR"

echo "CONFIG_PATH: $CONFIG_PATH"
echo "ROLL_PATH:   $ROLL_PATH"
echo "ROLL_OUTPUT_DIR: $ROLL_OUTPUT_DIR"

# ── Defensive libcuda preload (triton 3.2.0 symbol-resolution insurance) ──
# Triton builds cuda_utils.so leaving the driver symbols (cuModuleGetFunction, ...)
# UND, expecting an already globally-loaded libcuda. deepspeed's triton ops can
# dlopen it during `import transformers` before torch loads libcuda, giving
#   ImportError: cuda_utils.so: undefined symbol: cuModuleGetFunction
# This bit the Midway *container* (weird staged libcuda). Natively it usually
# resolves (torch loads the one real driver libcuda first), but preloading it is
# harmless insurance against the import-order race. Unlike the container path,
# there is no /.singularity.d/libs here — find the host driver libcuda directly.
HOST_LIBCUDA=$(ldconfig -p 2>/dev/null | awk '/libcuda\.so\.1/{print $NF; exit}')
[ -z "$HOST_LIBCUDA" ] && HOST_LIBCUDA=$(ls /usr/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null | head -1)
if [ -n "$HOST_LIBCUDA" ]; then
    export LD_PRELOAD="${HOST_LIBCUDA}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "LD_PRELOAD=$LD_PRELOAD (host driver libcuda for triton symbol resolution)"
else
    echo "NOTE: no libcuda.so.1 found for preload (expected on a login node; should be present on a compute node)"
fi

# ── Cap per-process CPU thread pools (REQUIRED on Polaris) ──
# This pipeline colocates 9 model workers on one node (4 actor_train + 4 actor_infer
# + 1 reference). By default each process's OpenBLAS/OMP pools spawn one thread per
# hardware thread (64 here), so 9×(64 BLAS + 64 OMP + torch/Ray/CUDA threads) blows
# past the PBS job's cgroup pids.max and workers die at construction with:
#   OpenBLAS blas_thread_init: pthread_create failed for thread N of 64:
#       Resource temporarily unavailable ... RLIMIT_NPROC 2060880 current, 2060880 max
# Note RLIMIT_NPROC (ulimit -u) is already ~2M — the wall is the cgroup thread cap,
# which we can't raise, so we shrink each process's pools instead (OpenBLAS even says
# "set a smaller OPENBLAS_NUM_THREADS"). Must be exported BEFORE python so every Ray
# worker inherits it before its first numpy/OpenBLAS import. See polaris_pbs_notes.md.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Cap Ray's CPU count so it prestarts ~16 idle workers instead of one per detected
# core (~64). Each idle worker carries ~30+ Ray threads, so the default pool is the
# bulk of the startup thread peak that crosses the job cgroup's pids.max=4096 and
# intermittently kills a critical actor (worker creation succeeded in one 1-GPU run
# and failed in the next — pure headroom variance). ROLL's start_ray_cluster()
# (initialize.py) reads RAY_NUM_CPUS and passes it to `ray start --num-cpus`. 16 is
# ample for our ~10 CPU actors while ~4x shrinking the idle pool.
export RAY_NUM_CPUS=16
# DeepSpeed JIT-builds fused_adam with ninja, which defaults to one compile job per
# core (64) -> a burst of ~64 concurrent gcc/nvcc subprocesses. Under the job cgroup's
# pids.max=4096 (see polaris_pbs_notes.md), that burst — concurrent with the Ray
# workers/schedulers initializing — re-triggers EAGAIN. Cap it. (Best paired with a
# pre-built/cached fused_adam in $TORCH_EXTENSIONS_DIR so no compile happens at all.)
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.0}
# Keep the vLLM V1 engine (the default). V1 is markedly LIGHTER on startup threads
# than V0 — under the job cgroup's pids.max=4096, V0 pushed the startup thread peak
# over the cap and killed the RolloutScheduler before training, whereas V1 reaches the
# training loop. The one V1 incompatibility — ROLL's per-parameter weight-sync passing
# a CUDA tensor through collective_rpc (which V1 msgpack-serializes across processes) —
# is fixed at the source in roll/third_party/vllm/vllm_0_8_4/llm.py:update_parameter
# (it now .cpu()s the tensor for V1). So we do NOT force V0.
export VLLM_USE_V1=1

python examples/start_agentic_pipeline.py \
  --config_path $CONFIG_PATH \
  --config_name agentic_val_tictactoe_selfplay_polaris_smoke \
  2>&1 | tee "$ROLL_LOG_DIR/custom_logs.log"
