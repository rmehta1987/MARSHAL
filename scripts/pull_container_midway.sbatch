#!/bin/bash
# One-shot pull of the official ROLL apptainer image on partition=build.
# Submit from the repo root:
#   sbatch scripts/pull_container_midway.sbatch
#
# Modelled on /project/rcc/mehta5/vllm/sbatch_create_sif.sh (the existing
# Decrypto-side pull script that worked). Image lands next to the existing
# vllm-v0.7.3.sif at /project/rcc/mehta5/vllm/.
#
#SBATCH --job-name=marshal-pull
#SBATCH --output=/project/rcc/mehta5/MARSHAL/logs/pull_%j.out
#SBATCH --error=/project/rcc/mehta5/MARSHAL/logs/pull_%j.err
#SBATCH --partition=build
#SBATCH --account=rcc-staff
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -e

mkdir -p /project/rcc/mehta5/MARSHAL/logs

module load apptainer/1.4.1

# Apptainer scratch + cache must live on /project. The defaults fall back
# to /tmp (too small on compute) and ~/.apptainer (home quota is tight),
# both of which fill up mid-pull on a multi-GB image.
export APPTAINER_TMPDIR=/project/rcc/mehta5/vllm/apptainer_tmp
export APPTAINER_CACHEDIR=/project/rcc/mehta5/vllm/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

cd /project/rcc/mehta5/vllm

echo "=== Pull job $SLURM_JOB_ID on $(hostname) at $(date) ==="
apptainer --version

time apptainer pull marshal_env_torch260_vllm084.sif \
    docker://roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084

echo "=== done ==="
ls -lh marshal_env_torch260_vllm084.sif
apptainer inspect marshal_env_torch260_vllm084.sif || true
