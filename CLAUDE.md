# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Why we're here

This is the **MARSHAL cluster bring-up on RCC Midway**, following the same playbook
already proven for the Decrypto port (see `/project/rcc/mehta5/decrypto/midway_notes.md`
for the template).

The immediate goal is **cluster bring-up only**: prove MARSHAL can run end-to-end on
Midway for one small smoke configuration. We are NOT reproducing paper results, NOT
evaluating the algorithm, and NOT tuning hyperparameters. Success = a training job
that starts, completes a handful of optimization steps without crashing, and writes
a checkpoint or log proving the loop closed.

The persistent record of fixes/decisions lives in `midway_notes.md` at this repo
root (author it on first run, mirroring the Decrypto one). The bring-up brief lives
at `/project/rcc/mehta5/marshal_bringup_prompt.md`.

## Cluster facts (Midway / RCC)

| Item | Value |
|---|---|
| Account | `rcc-staff` |
| Partition available | `test` only (no `sinfo` exploration — we only have this one) |
| GPU target | NVIDIA H200, constraint `H200` |
| Per-node | 4× H200, ~140 GiB VRAM each |
| Driver | 535.216.03 (max CUDA 12.2) — torch must work via CUDA Minor-Version Compatibility |
| Python module | `python/miniforge-25.3.0` |
| Activation | `module load python/miniforge-25.3.0 && eval "$(mamba shell hook --shell bash)" && mamba activate <env>` (NOT `source activate` — falls through to system Python 3.8) |
| Project root | `/project/rcc/mehta5/MARSHAL` |
| Existing rollout env | `/project/rcc/mehta5/conda-envs/vllm-probe` (vllm 0.10.2, torch 2.8.0+cu128, transformers<5) — likely reusable for rollouts |
| Training env (to create) | `/project/rcc/mehta5/conda-envs/marshal-train` (RL libs usually need different torch/cuda) |
| Model cache | `/project/rcc/mehta5/vllm/models/` (Qwen2.5-72B-Instruct, Meta-Llama-3.1-70B-Instruct already present) |
| HF cache | `/project/rcc/mehta5/hf_cache` |
| Inductor cache | `/project/rcc/mehta5/torchinductor_cache` |
| Smoke wall-time | default `--time=02:00:00` (test partition caps below 12h) |

## Lessons inherited from the Decrypto port — apply these, don't relearn them

1. **TMPDIR hygiene** — `--export=ALL` propagates the submitting shell's `TMPDIR`
   (often pointing at a now-cancelled job's scratch). In every sbatch wrap:
   ```bash
   unset TMPDIR SLURM_TMPDIR
   export TMPDIR=/tmp/${USER}_${SLURM_JOB_ID}
   mkdir -p $TMPDIR
   ```
2. **`transformers` must be pinned `<5`** — 5.x removed `all_special_tokens_extended`
   which vllm 0.10.2 still calls. Pair with `tokenizers<0.22`.
3. **Inductor cache lives in /project** — set
   `TORCHINDUCTOR_CACHE_DIR=/project/rcc/mehta5/torchinductor_cache` so torch.compile
   doesn't try to write into transient SLURM scratch.
4. **`--enforce-eager` for smoke tests** — skips torch.compile entirely. Use until
   the basic loop works, then revisit for perf.
5. **Don't `pip install -r requirements*.txt` blindly.** MARSHAL ships requirement
   files pinning `torch==2.6.0` / `vllm==0.8.4` (or 2.5.1 / 0.7.3). The driver-535
   cluster runs torch 2.8.0+cu128 / vllm 0.10.2 via CUDA Minor-Version Compatibility.
   Inspect first, install incrementally, expect to deviate from the pins.
6. **Conda activation** — see the cluster-facts table above. Always use the mamba
   pattern.

## Repo architecture (big picture)

MARSHAL is **built on top of the [ROLL framework](https://github.com/alibaba/ROLL)**.
The `roll/` directory at this repo's root is a vendored copy of ROLL — the training
framework, not MARSHAL-specific logic. MARSHAL's contribution is the algorithm
(Turn-level Advantage Estimator + Agent-specific Advantage Normalization) plus the
game environments and pipeline configs.

Key directories:

- `roll/` — the ROLL training framework (pipelines, workers, models, distributed).
  Two pipeline families live under `roll/pipeline/`: `agentic/` (multi-turn self-play,
  what MARSHAL uses) and `rlvr/` (RL with verifiable rewards, the original ROLL
  use case). `roll/agentic/env/` and `roll/agentic/rollout/` are the multi-agent
  episode machinery.
- `examples/<game>/` — one directory per game (tictactoe, connect_four, kuhn_poker,
  leduc_poker, hanabi, multi_games). Each contains a hydra `agentic_val_<game>_*.yaml`
  config and a `run_agentic_pipeline_<game>_*.sh` launcher.
- `examples/config/` — shared hydra fragments (`envs.yaml`, `deepspeed_zero{,2,3,_cpuoffload}.yaml`)
  that the per-game configs inherit via `defaults:`.
- `examples/start_agentic_pipeline.py` — the actual Python entry point. Per-game
  shell scripts call this with `--config_path <game-dir> --config_name <yaml-stem>`.
- `mcore_adapter/` — a local Megatron-Core adapter package, installed via the `./mcore_adapter`
  line in `requirements_common.txt`. Configs that use `strategy_name: megatron_train`
  require this.
- `scripts/` — repo-provided slurm wrapper (`train.sbatch`) and model-conversion
  utilities. **`scripts/train.sbatch` targets the original CMU `ycleong` cluster**
  (`/net/projects2/ycleong/sg/...`, `--partition=general`, `a100|h100|h200`,
  `--time=12:00:00`, apptainer container at `/net/projects2/ycleong/sg/containers/marshal_env`).
  Author Midway equivalents (e.g. `scripts/train_midway.sbatch`) instead of editing in place,
  so the original diff remains visible.
- `docker/` — Dockerfiles for four toolchain combos (torch 2.5.1 / 2.6.0 × sglang / vllm).
  Mirrors the four `requirements_torch*_{vllm,sglang}.txt` files. Inform env-build
  choices but we don't build images on Midway.

Self-play training pipeline at a glance:
1. Ray cluster is started (the per-game launcher does aggressive `ray stop`/cleanup
   before launching — those preambles are deliberate, keep them).
2. `start_agentic_pipeline.py` reads the hydra config, which has three model roles:
   `actor_train` (e.g. `megatron_train` strategy), `actor_infer` (`vllm` strategy
   for rollouts), and `reference` (`hf_infer` strategy for KL reference). All three
   point at the same `pretrain:` model by default — e.g. `Qwen/Qwen3-4B` for tictactoe.
3. Rollouts are generated via vLLM in the configured environments
   (`custom_envs:` block — uses OpenSpiel/pyspiel under the hood).
4. The training loop uses `adv_estimator: reinforce` + `advantage_norm: mean`
   (configurable) — these are MARSHAL's contribution.
5. Logs/checkpoints land in `$ROLL_OUTPUT_DIR` (`results/<exp>/<timestamp>/`).
6. Tracking via `wandb` (needs `WANDB_API_KEY`); switchable to tensorboard in YAML.

## Common commands

```bash
# Activate the rollout env (likely reused; training env may need to be separate)
module load python/miniforge-25.3.0
eval "$(mamba shell hook --shell bash)"
mamba activate /project/rcc/mehta5/conda-envs/vllm-probe

# OpenSpiel / pyspiel is required by MARSHAL environments
pip install pyspiel

# Launch a per-game self-play pipeline directly (interactive — for debugging only)
cd /project/rcc/mehta5/MARSHAL
export PYTHONPATH="$PWD:$PYTHONPATH"
bash examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay.sh

# Rollout-only (no gradient updates) — smallest smoke shape
bash examples/tictactoe/run_agentic_rollout_tictactoe.sh

# Run the test suite
make test            # python -m pytest -n auto --dist=loadfile -s -v ./tests/

# Lint / format
make precommit       # pre-commit run --all-files
# black / ruff config in pyproject.toml: line-length 119, target py310
```

The training entry script is:

```bash
python examples/start_agentic_pipeline.py \
  --config_path <game-folder-name> \
  --config_name <yaml-stem-no-extension>
```

The `--config_path` is resolved relative to `examples/` by the per-game launcher
(`CONFIG_PATH=$(basename $(dirname $0))`), so launcher and config must live in the
same `examples/<game>/` dir.

## Slurm — porting the inherited `scripts/train.sbatch`

The shipped `scripts/train.sbatch` will NOT run on Midway as-is. When authoring the
Midway version, apply (at minimum) the same surgical edits used for Decrypto:

- `--partition=general` → `--partition=test`; add `--account=rcc-staff`
- `--constraint="a100|h100|h200"` → `--constraint=H200`
- `--time=12:00:00` → `--time=02:00:00` (test partition cap)
- Replace `/net/projects2/ycleong/sg/...` paths with `/project/rcc/mehta5/MARSHAL/...`
- Remove the `apptainer exec ... $CONTAINER_PATH` wrap — we don't have that container
  image on Midway. Activate the mamba env in the wrap instead.
- Apply the TMPDIR-hygiene + `TORCHINDUCTOR_CACHE_DIR` + `--enforce-eager` fixes
  from the lessons-learned list above.
- For smoke runs, pick the smallest model the configs support (Qwen3-4B is what
  the tictactoe selfplay config defaults to) on 1–2 GPUs. The model is not in
  the local cache yet — either let HF download it (set HF cache vars) or
  pre-download with `huggingface-cli`.
- The shipped sbatch wires `signal=B:SIGUSR1@90` + `train_autoresume.sh` for
  auto-resume on time-limit. Keep that pattern; it works fine on Midway.

## Done criteria

A green "MARSHAL smoke test" entry in `midway_notes.md` citing the SLURM jids,
elapsed wallclock, and the artifact (log line, checkpoint, summary file) that
proves the training loop closed — same shape as the GREEN entries in the
Decrypto midway_notes.md.
