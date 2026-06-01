# Running MARSHAL on RCC Midway — Setup & Run Guide

This is a practical, start-to-finish guide for installing and running MARSHAL
(self-play RL on top of the ROLL framework) on the RCC **Midway** cluster. It
distills the hard-won bring-up record in `midway_notes.md` into the steps you
actually need to follow. If something here surprises you, `midway_notes.md` has
the blow-by-blow reasoning behind every decision.

The short version: **we run inside the official ROLL Apptainer container**, not
a hand-built conda env. The container carries the exact version-locked stack
ROLL was built against, which sidesteps a dependency deadlock you cannot solve
otherwise (details below). All the cluster-specific glue lives in three kinds of
files per experiment: a hydra **config**, a **launcher** script, and an **sbatch**
wrapper.

---

## 1. The cluster, in one table

| Item | Value |
|---|---|
| Account | `rcc-staff` |
| Partition | `test` (the only one we have; ≤12 h jobs, we use 2 h) |
| GPUs | 4× NVIDIA **H200** per node (~140 GiB each), constraint `H200` |
| Host driver | 535.216.03 → max CUDA 12.2. Newer-CUDA containers work via CUDA **Minor-Version Compatibility** (MVC) |
| Apptainer | `module load apptainer/1.4.1` |
| Project root | `/project/rcc/mehta5/MARSHAL` |
| Container image | `/project/rcc/mehta5/vllm/marshal_env_torch260_vllm084.sif` (torch 2.6.0+cu124, vllm 0.8.4, ray 2.46.0, deepspeed 0.16.4, megatron-core 0.12.3, transformer-engine 2.2.0, flash-attn 2.7.2) |
| Model store | `/project/rcc/mehta5/vllm/models/` (Qwen2.5-0.5B-Instruct, Qwen3-4B, Qwen2.5-72B, Llama-3.1-70B) |
| HF cache | `/project/rcc/mehta5/hf_cache` |
| Triton / Inductor caches (container) | `/project/rcc/mehta5/triton_cache_container`, `/project/rcc/mehta5/torchinductor_cache_container` |

> **Home and scratch are too small or transient.** Keep models, caches, and
> outputs on `/project`.

---

## 2. Why the container (and not a conda env)

We first tried building a `marshal-train` conda env from MARSHAL's requirement
files. It dies on an unwinnable version conflict: ROLL pins `ray<=2.46.0` (its
log-monitor uses an old Ray API), but **vllm 0.10.2 requires `ray>=2.48`** — no
single Ray version satisfies both. You can patch around it for a while, but it's
whack-a-mole.

The official ROLL image already contains a *consistent* ray/vllm/deepspeed/
megatron stack, so we run everything inside it with `apptainer exec --nv`. The
`.sif` is read-only, which forces two small accommodations you'll see below
(packages layered via `PYTHONPATH` instead of `pip install`).

---

## 3. One-time setup

If the image and models are already on disk (they are, as of this writing), you
can skip straight to **Section 5 — Running a job**. These are the steps to
reproduce the setup from scratch.

### 3a. Pull the container image

```bash
cd /project/rcc/mehta5/MARSHAL
sbatch scripts/pull_container_midway.sbatch     # runs on partition=build (has internet)
```

This pulls `docker://.../roll/pytorch:nvcr-24.05-py3-torch260-vllm084` into
`/project/rcc/mehta5/vllm/marshal_env_torch260_vllm084.sif` (~10 GB). The script
points Apptainer's scratch/cache at `/project` so the multi-GB pull doesn't fill
`/tmp` or your home quota.

### 3b. Download a model

Models live as flat snapshots under `/project/rcc/mehta5/vllm/models/<name>`.
The login node has internet access to Hugging Face, and the `vllm-probe` conda
env ships the `huggingface-cli`:

```bash
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/rcc/mehta5/conda-envs/vllm-probe
export HF_HOME=/project/rcc/mehta5/hf_cache

huggingface-cli download Qwen/Qwen3-4B \
    --local-dir /project/rcc/mehta5/vllm/models/Qwen3-4B \
    --exclude "*.pth" "*.gguf" "original/*"
```

Qwen2.5-0.5B-Instruct (the cheapest smoke model) and Qwen3-4B (the config
family's default) are already present.

### 3c. The `container_extras/` directory

A couple of Python packages aren't in the image and can't be `pip install`-ed
into a read-only `.sif`, so they're installed into a directory that gets put on
`PYTHONPATH` instead:

```bash
mkdir -p /project/rcc/mehta5/MARSHAL/container_extras
# example: open_spiel / pyspiel for the game environments, if not already in the image
# pip install --target /project/rcc/mehta5/MARSHAL/container_extras open_spiel
```

(`import pyspiel` comes from the `open_spiel` package — the MARSHAL README's
`pip install pyspiel` is wrong; that PyPI name doesn't exist.)

---

## 4. How a run is wired together

Each experiment is three files. To make a new experiment, copy a matching set
and edit. **Don't edit the proven ones in place** — author parallel copies so
the working baselines stay intact (this mirrors the repo's existing convention).

```
examples/tictactoe/
  agentic_val_tictactoe_selfplay_midway_smoke.yaml      # hydra config  (0.5B, deepspeed, 3 steps)   ← GREEN baseline
  agentic_val_tictactoe_selfplay_midway_scaleup.yaml    # Qwen3-4B, deepspeed, 20 steps
  agentic_val_tictactoe_selfplay_midway_megatron.yaml   # Qwen3-4B, megatron_train TP=4, 3 steps
  run_agentic_pipeline_tictactoe_selfplay_midway*.sh    # launcher (runs INSIDE the container)
scripts/
  train_midway.sbatch            # SLURM wrapper for the smoke
  train_midway_scaleup.sbatch    # SLURM wrapper for the scale-up
  train_midway_megatron.sbatch   # SLURM wrapper for the megatron run
```

**The sbatch** requests the node (4× H200, `test`, 2 h), sets up the
environment, and calls `apptainer exec --nv ... <launcher>`. It binds `/project`
and `/scratch` into the container, forces a per-job `TMPDIR`, points HF/Triton/
Inductor caches at `/project`, and sets the container `PYTHONPATH`.

**The launcher** runs *inside* the container. It does aggressive Ray cleanup
(stale Ray sessions from prior jobs poison a new head node), resolves the hydra
config path, applies the runtime fixes from Section 6, and finally calls
`python examples/start_agentic_pipeline.py --config_path tictactoe --config_name <stem>`.

**The config** is standard ROLL hydra. It defines three model roles, all
pointing at the same `pretrain:` model:
- `actor_train` — the policy being trained (`deepspeed_train` or `megatron_train`)
- `actor_infer` — rollout generation (`vllm`)
- `reference` — KL reference (`hf_infer`)

plus the game env (`custom_envs`), batch sizes, and the MARSHAL algorithm knobs
(`adv_estimator: reinforce`, `advantage_norm: mean`, turn-level scoring).

---

## 5. Running a job

From the repo root:

```bash
cd /project/rcc/mehta5/MARSHAL

# the proven smoke (0.5B, deepspeed, 3 steps) — closes the loop in ~7 min
sbatch scripts/train_midway.sbatch

# a realistic model (Qwen3-4B, deepspeed, 20 steps) — ~21 min
sbatch scripts/train_midway_scaleup.sbatch

# the megatron path (Qwen3-4B, TP=4 sequence-parallel, 3 steps) — ~16 min
sbatch scripts/train_midway_megatron.sbatch
```

Watch it:

```bash
squeue -u $USER
tail -f logs/train_midway_<jobid>.out
```

Output lands in `results/<experiment>/<jobid>_<timestamp>/`:
- `logs/` — per-rank logs + the combined `custom_logs.log`
- `tensorboard/` — scalar metrics per step (`actor/pg_loss`, `critic/score/mean`,
  `tokens/response_length`, `env/TicTacToe/...`, etc.)
- `actor_train-*/checkpoint-N/` — the model checkpoints
- `pipeline/checkpoint-N/` — pipeline bookkeeping

**Checkpoints are large** (deepspeed: ~3 GB/rank; megatron: ~14 GB/rank). For
smoke/validation runs, once you've confirmed success you can delete the heavy
`actor_train-*/` dirs and keep `tensorboard/` + `logs/` as proof — they're a
few MB and contain the per-step metrics.

---

## 6. The fixes baked into the launchers/sbatch (and why)

These are non-obvious and easy to lose. They're already applied in the files
above; this is so you understand them and carry them into new experiments.

**TMPDIR hygiene (sbatch).** `--export=ALL` propagates the submitting shell's
`TMPDIR`, often pointing at a cancelled job's scratch. Every wrap does:
```bash
unset TMPDIR SLURM_TMPDIR
export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}; mkdir -p "$TMPDIR"
```
and binds that path into the container so the inner process sees the same one.

**`mcore_adapter` on PYTHONPATH (sbatch).** The image can't be `pip install
-e ./mcore_adapter`'d (read-only), and `vllm_strategy.py` imports
`mcore_adapter.models.*`. So the sbatch sets
`PYTHONPATH=...container_extras:...mcore_adapter/src`. Without it, the
bind-mounted repo's `mcore_adapter/` is picked up as a namespace package whose
submodules don't resolve.

**`LD_PRELOAD` the host libcuda (every launcher).** Triton 3.2.0 builds its
`cuda_utils.so` *without* linking `libcuda.so.1` — the driver symbols
(`cuModuleGetFunction`, …) are left undefined, expected to resolve from a
globally-loaded driver. DeepSpeed's Triton ops trigger that load during
`import transformers`, *before* torch has loaded libcuda, so you get:
```
ImportError: .../cuda_utils.so: undefined symbol: cuModuleGetFunction
```
Fix: preload the 535 host driver libcuda that `apptainer --nv` stages into
`/.singularity.d/libs/`:
```bash
HOST_LIBCUDA=$(ls /.singularity.d/libs/libcuda.so.1 2>/dev/null || ls /.singularity.d/libs/libcuda.so* 2>/dev/null | head -1)
[ -n "$HOST_LIBCUDA" ] && export LD_PRELOAD="${HOST_LIBCUDA}${LD_PRELOAD:+:$LD_PRELOAD}"
```

**Megatron only — route TransformerEngine off cuDNN (`*_megatron.sh`).** TE's
default cuDNN fused-attention backend has no execution plan for Qwen3's
attention graph on this image's cuDNN over the 535/CUDA-12.2 driver, and dies
with `cuDNN Error: No execution plans support the graph`. Send TE through
flash-attn (which uses its own kernels) instead:
```bash
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
```

**Megatron only — CUDA work-queue ordering (`*_megatron.sh`).** With
`tensor_model_parallel_size>1` + `sequence_parallel`, Megatron needs a single
CUDA work queue for correct TP overlap. ROLL doesn't set it:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**`set -o pipefail` (megatron launcher; recommended everywhere).** The launchers
end with `python ... | tee logfile`. Without `pipefail`, the script returns
*tee's* exit code (always 0), so a crashed run reports `sacct` **COMPLETED 0:0**
and prints "Training exited with code: 0". Add `set -o pipefail` near the top so
the real Python exit status propagates and `sacct` tells the truth.

> **Note:** the deepspeed launchers (`run_..._midway.sh`,
> `run_..._scaleup.sh`) don't yet have `set -o pipefail`. It hasn't caused a
> wrong call because those runs genuinely succeeded — but add it if you want
> their `sacct` status to be trustworthy on a future failure.

---

## 7. Making your own config

Start from the closest working config and change as little as possible. The
knobs that matter:

| Knob | Where | Notes |
|---|---|---|
| Model | `pretrain:` | Use a local path under `/project/rcc/mehta5/vllm/models/`. Download first (3b). |
| Chat template | `data_args.template` (all 3 roles) | Must match the model: `qwen2_5` for Qwen2.5, `qwen3` for Qwen3. Mismatched templates tokenize wrong. |
| Steps | `max_steps`, `save_steps` | `save_steps == max_steps` writes one checkpoint at the end. |
| Train strategy | `actor_train.strategy_args.strategy_name` | `deepspeed_train` (+ `${deepspeed_zero2}` config) or `megatron_train` (+ TP/SP config). |
| Tensor parallel | megatron `tensor_model_parallel_size` | **`num_attention_heads` must be divisible by TP.** Qwen3-4B has 32 heads (TP=4 ok); Qwen2.5-0.5B has 14 (TP=4 fails — use TP=1 or 2, or a different model). |
| Attention | `model_args` | deepspeed path: `attn_implementation: eager`. megatron path: `flash_attn: fa2` (and the NVTE env vars from §6). |
| vLLM memory | `actor_infer.strategy_config.gpu_memory_utilization` | 0.5–0.8. Lower if the colocated train/reference roles OOM. |
| Sequence length | `sequence_length` | With megatron `sequence_parallel`, must be divisible by TP. |

When you make a new config, also copy its launcher and sbatch (swap the
`--config_name`, the `--job-name`, and the `results/<experiment>` label), so the
output of different experiments doesn't collide.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cuda_utils.so: undefined symbol: cuModuleGetFunction` at `import transformers` | Triton built without libcuda; symbols not in global namespace yet | `LD_PRELOAD` the host libcuda (§6). If a stale cache is suspected, `rm -rf /project/rcc/mehta5/triton_cache_container/*`. |
| `cuDNN Error: No execution plans support the graph` (TransformerEngine) | TE cuDNN fused attention unsupported for this attn graph on this driver | `NVTE_FUSED_ATTN=0` + `NVTE_FLASH_ATTN=1` (§6). Megatron path only. |
| `ModuleNotFoundError: No module named 'mcore_adapter.models'` | repo's `mcore_adapter/src` not on PYTHONPATH | Confirm the sbatch sets `PYTHONPATH=...mcore_adapter/src` (§6). |
| Megatron asserts on `num_attention_heads` / TP | heads not divisible by `tensor_model_parallel_size` | Lower TP or pick a model whose head count divides evenly (§7). |
| `rm: cannot remove '/usr/local/cuda/compat/lib': Read-only file system` (in `.err`) | image startup tries to drop its bundled compat lib; the `.sif` is read-only | **Harmless noise** — compat isn't on the load path. Ignore. |
| Job says COMPLETED / "exited 0" but training clearly failed | `python ... | tee` masks the exit code | Add `set -o pipefail` to the launcher (§6); re-check `sacct`. |
| Ray init errors / stale head node | leftover Ray session from a prior job | The launcher's `ray stop`/`pkill` preamble handles this; keep it. |
| `pip install -r requirements*.txt` fixes nothing / conflicts | the shipped torch/vllm/ray pins are wrong for this cluster | Don't. Use the container. |

---

## 9. What "success" looks like

A run that closed the loop will show, in `logs/train_midway_<jobid>.out`:
- `Training exited with code: 0` (trustworthy only with `pipefail`),
- real optimizer metrics in the per-step JSON: `actor/pg_loss`, `actor/kl_loss`,
  `actor_train/grad_norm`, `critic/score/mean`, `system/step`, `system/tps`,
- a checkpoint saved at the final step:
  - deepspeed: `actor_train-*/checkpoint-N/checkpoint/bf16_zero_pp_rank_*_optim_states.pt`
  - megatron: `actor_train-*/checkpoint-N/iter_*/mp_rank_*/model_optim_rng.pt`

Both training strategies (**deepspeed_train** and **megatron_train**) are
confirmed working in the container on Midway. See the GREEN entries in
`midway_notes.md` for reference job IDs, wallclocks, and exact artifacts.
