#!/bin/bash
# Midway-flavored MARSHAL training launcher (container path).
# Wraps the per-game launcher in `apptainer exec --nv` against the official
# ROLL image, which carries the version-locked stack ROLL was built against
# (torch 2.6.0+cu124, vllm 0.8.4, ray 2.46.0, deepspeed 0.16.4, mcore_adapter).
# The earlier source-install marshal-train env hit a vllm/ray version impasse;
# see midway_notes.md for the full chain that led to switching.
#
# Submit from the MARSHAL repo root:
#   sbatch scripts/train_midway.sbatch
#
#SBATCH --job-name=marshal-smoke
#SBATCH --output=/project/rcc/mehta5/MARSHAL/logs/train_midway_%j.out
#SBATCH --error=/project/rcc/mehta5/MARSHAL/logs/train_midway_%j.err
#SBATCH --partition=test
#SBATCH --account=rcc-staff
#SBATCH --constraint=H200
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=16G

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date) ==="

PROJECT_ROOT="/project/rcc/mehta5/MARSHAL"
IMAGE="/project/rcc/mehta5/vllm/marshal_env_torch260_vllm084.sif"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# ── TMPDIR hygiene ──
# --export=ALL propagates the submitting shell's TMPDIR (often a now-cancelled
# job's scratch). Force a per-job TMPDIR on the host; bind it into the
# container so the inner process sees the same path.
unset TMPDIR SLURM_TMPDIR
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"

# ── Caches on /project (home/scratch are too small or transient) ──
export HF_HOME=/project/rcc/mehta5/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME
# Container-scoped triton/inductor caches. The earlier source-install path
# poisoned /project/rcc/mehta5/triton_cache with cuda_utils.so symbols that
# don't resolve under the container's libcuda (undefined symbol
# cuModuleGetFunction at import time). Keep them separate per code path.
export TRITON_CACHE_DIR=/project/rcc/mehta5/triton_cache_container
export TORCHINDUCTOR_CACHE_DIR=/project/rcc/mehta5/torchinductor_cache_container
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# ── Ray scratch ──
export RAY_TMPDIR="$TMPDIR/ray"
mkdir -p "$RAY_TMPDIR"

# ── Output dir ──
export ROLL_OUTPUT_DIR="$PROJECT_ROOT/results/tictactoe_selfplay_midway_smoke/${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
export ROLL_LOG_DIR="$ROLL_OUTPUT_DIR/logs"
export ROLL_RENDER_DIR="$ROLL_OUTPUT_DIR/render"
mkdir -p "$ROLL_LOG_DIR" "$ROLL_RENDER_DIR"

# ── Container PYTHONPATH ──
# Repo gets prepended by the launcher (cd'd to PROJECT_ROOT). Two extras:
# - container_extras/: pip-target dir for packages we layer on top of the
#   read-only sif (open_spiel → pyspiel).
# - mcore_adapter/src/: vendored mcore_adapter package source. Normally
#   installed via `pip install -e ./mcore_adapter`, but the sif is read-only
#   so we put it on PYTHONPATH instead. Without this, the bind-mounted repo's
#   `mcore_adapter/` directory gets picked up as a namespace package whose
#   submodules don't resolve (vllm_strategy.py imports mcore_adapter.models.*).
export CONTAINER_PYTHONPATH="$PROJECT_ROOT/container_extras:$PROJECT_ROOT/mcore_adapter/src"

# ── Module + image checks ──
source /etc/profile.d/modules.sh 2>/dev/null || true
module load apptainer/1.4.1
[ -f "$IMAGE" ] || { echo "ERROR: image not found at $IMAGE"; exit 1; }

echo "=== Host env ==="
nvidia-smi | head -20
echo "Image: $IMAGE ($(stat -c %s "$IMAGE") bytes)"

# ── Cleanup on exit ──
# Ray procs spawn in the container's mount ns but share host PID ns (no --pid),
# so host-side pkill finds them. ray stop --force must run inside the image
# (needs the python entry point).
cleanup() {
    echo "$(date): Cleanup triggered (exit)"
    apptainer exec --nv \
        --bind /project,/scratch \
        --bind "$TMPDIR:$TMPDIR" \
        --env RAY_TMPDIR="$RAY_TMPDIR" \
        "$IMAGE" \
        bash -c "ray stop --force 2>/dev/null" || true
    sleep 2
    pkill -u "$(whoami)" -f "ray::"      2>/dev/null || true
    pkill -u "$(whoami)" -f "raylet"     2>/dev/null || true
    pkill -u "$(whoami)" -f "gcs_server" 2>/dev/null || true
    rm -rf /tmp/ray/session_latest 2>/dev/null || true
    echo "$(date): Cleanup complete"
}
trap cleanup EXIT

# ── Run training inside the container ──
# --nv pulls in the host NVIDIA driver libs; --bind /project,/scratch covers
# repo + caches + model store. We explicitly pass the env vars the inner
# process needs (apptainer otherwise filters most through, but being explicit
# avoids drift if the user's submitting shell sets something surprising).
apptainer exec --nv \
    --bind /project,/scratch \
    --bind "$TMPDIR:$TMPDIR" \
    --env "TMPDIR=$TMPDIR" \
    --env "LD_LIBRARY_PATH=/.singularity.d/libs:/usr/local/cuda/lib64:/usr/local/nvidia/lib64" \
    --env "TRITON_LIBCUDA_PATH=/.singularity.d/libs/libcuda.so.1" \
    --env "PYTHONPATH=$CONTAINER_PYTHONPATH" \
    --env "HF_HOME=$HF_HOME" \
    --env "HUGGINGFACE_HUB_CACHE=$HUGGINGFACE_HUB_CACHE" \
    --env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" \
    --env "TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR" \
    --env "RAY_TMPDIR=$RAY_TMPDIR" \
    --env "ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR" \
    --env "ROLL_LOG_DIR=$ROLL_LOG_DIR" \
    --env "ROLL_RENDER_DIR=$ROLL_RENDER_DIR" \
    --env "WANDB_API_KEY=${WANDB_API_KEY:-}" \
    --env "SLURM_JOB_ID=$SLURM_JOB_ID" \
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-32}" \
    "$IMAGE" \
    bash -c "cd $PROJECT_ROOT && bash examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_midway.sh"
TRAIN_EXIT=$?
echo "Training exited with code: $TRAIN_EXIT"
exit $TRAIN_EXIT
