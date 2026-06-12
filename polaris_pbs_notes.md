# MARSHAL Polaris (PBS) port — running notes

Living doc. Tracks the ALCF **Polaris** (PBS Pro) bring-up of MARSHAL, the PBS
analog of the RCC Midway (Slurm) port. The template and the load-bearing fixes
come from `midway_notes.md`; read that first. Authored per
`polaris_handoff_prompt.md`.

Scope is **cluster bring-up only**: one small smoke-config training run that
completes a handful of optimization steps and writes an artifact. NOT paper
repro, NOT eval, NOT hyperparameter tuning.

> **Headline divergence from Midway/the handoff:** the handoff assumed the
> official ROLL **Apptainer container** is the unit of portability. On Polaris
> that path is blocked — there is no `.sif` here, the Aliyun ROLL registry is
> unreachable from ALCF, and `container_extras/` is `.gitignore`'d (so a fresh
> clone doesn't carry it). We therefore went the **native conda/venv** route
> instead, installing ROLL's *exact pinned stack* (the same versions the
> container froze) so we sidestep the ray/vllm impasse that killed the Midway
> source-install. See "Env strategy" below.

---

## STATUS (2026-06-12): deepspeed smoke bring-up complete (GREEN, 2026-06-06); megatron scale-up in progress

The 0.5B deepspeed smoke is complete and reproducible (see below). The megatron
scale-up (Qwen3-4B, `megatron_train` TP=4) is the current work item — see the
**"Megatron scale-up"** section and the **job ledger** further down.

The MARSHAL tictactoe self-play smoke ran end-to-end on Polaris (jid **7186746**): 3 DeepSpeed
REINFORCE steps → `pipeline complete!` → 12 G `checkpoint-2` + TensorBoard, `Training exited with
code: 0`. Full details + artifact paths in the **"GREEN — MARSHAL smoke test"** section below;
the 10-layer fix stack and every failed attempt are in the decisions log.

```bash
# Reproduce:
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/MARSHAL
qsub -v MARSHAL_VENV_TARBALL=/lus/eagle/projects/lighthouse-uchicago/members/mehta5/marshal-train-venv.tar \
     scripts/train_polaris.pbs
# Watch logs/wrap_<jid>.log + results/.../<jid>_*/logs/custom_logs.log for
# "pipeline complete!" and "Training exited with code: 0".
```

> Known residual flakiness: the startup EAGAIN race vs the debug-node cgroup `pids.max=4096`
> loses ~half the time at RolloutScheduler creation, and a lost run HANGS (qdel + resubmit).
> The durable fix is an ALCF ticket to raise the per-job `pids.max`. Everything else is fixed
> in-repo and reproducible.

---

## Cluster facts (Polaris / ALCF)

| Item | Value |
|---|---|
| Scheduler | **PBS Pro** (`qsub`/`qstat`/`qdel`), login host `polaris-login-02` |
| Account (`-A`) | **`lighthouse-uchicago`** — NOT "Uchicago-lighthouse" (PBS rejects it); confirmed via `sbank-list-allocations` (alloc 12374, ~17,184 node-h available) |
| Queue | `debug` (1–2 nodes, ≤1 h walltime) — used for the smoke. Also `debug-scaling` (1–10 nodes, 1 job/user), `prod` (routing, ≥10 nodes, 24 h) |
| GPU | 4× NVIDIA **A100 40 GiB** (HBM2, sm_80) per node — far tighter than Midway's H200 (~140 GiB); re-confirmed on-node 2026-06-12 (decrypto job 7197265, see the sibling port's notebook `../decrypto/polaris_pbs_notes.md`); there is no 80 GB partition |
| CPU / RAM | AMD EPYC Milan 7543P, 32c/64t (`ncpus=64`), 512 GiB DDR4 |
| Node-local scratch | pair of 1.6 TB SSDs in RAID0 (used for `TMPDIR`; mount assumed `/local/scratch`, wrap falls back to `/tmp`) |
| Native CUDA | **12.4.1** (`/soft/compilers/cudatoolkit/cuda-12.4.1`, ALCF's PyTorch is built against it). Base conda ships torch 2.8.0 (cu128) → driver supports ≥ CUDA 12.8, so our cu124 wheels are safe |
| Filesystems | `home`, `eagle` (`/lus/eagle/projects`), `grand`. Jobs **must** declare `-l filesystems=home:eagle` or PBS rejects them |
| Project root | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5/MARSHAL` |
| Member base | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` (venv, models, caches live here) |
| Training venv | `…/members/mehta5/conda-envs/marshal-train` (clean `python -m venv`, see below) |
| Model store | `…/members/mehta5/models/` (`Qwen2.5-0.5B-Instruct` staged) |
| HF / triton / inductor caches | `…/members/mehta5/{hf_cache,triton_cache,torchinductor_cache}` |
| Apptainer | 1.3.6 binary exists under `/soft/spack/testing/0.8.1/apptainer/...` (its module is broken on stale spack deps). **Unused** — we run native, not in a container |

### PBS submit idiom (single GPU node)

```bash
qsub -A lighthouse-uchicago -q debug \
     -l select=1:ncpus=64:ngpus=4 -l filesystems=home:eagle \
     -l walltime=01:00:00 scripts/train_polaris.pbs
# (these are baked into the #PBS header of scripts/train_polaris.pbs)
```

Note: `-l select=` does **not** take `:system=polaris`. `$PBS_JOBID` looks like
`7185571.polaris-pbs-01.…`; `${PBS_JOBID%%.*}` gives the clean numeric tag.

---

## Env strategy — native venv with ROLL's pinned stack (not a container)

The handoff's container path is unavailable on Polaris (no `.sif`, Aliyun
unreachable, `container_extras/` absent). The Midway notes also record that a
*source-install* env was tried there and abandoned over an unwinnable conflict:
ROLL pins `ray<=2.46.0`, but the torch-2.8/vllm-0.10.2 stack Midway's driver
forced needs `ray>=2.48`. **We dodge that here by construction**: install ROLL's
own pinned, mutually-consistent versions — the exact set the container froze —
where `vllm 0.8.4` is perfectly happy with `ray 2.46.0`.

```
torch 2.6.0+cu124 · vllm 0.8.4 · ray 2.46.0 · deepspeed 0.16.4 ·
transformers 4.51.2 · tokenizers 0.21.4 · numpy 1.26.4 · open_spiel 1.6.15
```

This works on Polaris because **cu124 is the native CUDA** (12.4.1) and the GPUs
are A100 sm_80 (the conda module even sets `TORCH_CUDA_ARCH_LIST=8.0`).

**Deliberate deviation from ALCF guidance:** the ALCF PyTorch docs recommend
*not* pip-installing a custom torch and using their base-conda torch (2.8.0)
instead. We must deviate — ROLL/vllm 0.8.4 pin `torch==2.6.0`. This is safe
because the smoke is **single-node** (Ray colocates all three roles on one
node's 4 GPUs over NVLink), so we need none of ALCF's multi-node AWS-OFI/NCCL
fabric (which their docs even warn can hang Megatron-DeepSpeed).

### Why the megatron/transformer-engine stack is *not* installed (for the smoke)

The smoke uses `deepspeed_train` + `attn_implementation: eager`, so it needs
neither `transformer-engine`, `megatron-core`, `apex`, nor `flash-attn`. Those
are deferred to the (optional) megatron path. One consequence is handled by a
source patch — see "RecvBucketManager stub" below.

---

## Env build — exact steps (2026-06-05)

All on the **login node** (has HF reachability + outbound; pip needs no GPU).
Caches pointed at eagle (home quota is small):

```bash
module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base   # base py 3.12.11
export PIP_CACHE_DIR=…/mehta5/pip_cache TMPDIR=…/mehta5/tmp
python -m venv …/mehta5/conda-envs/marshal-train        # clean (NO --system-site-packages)
source …/conda-envs/marshal-train/bin/activate
pip install --upgrade pip wheel setuptools

# 1) torch first (verifies the cu124 wheel): torch 2.6.0 → cuda build 12.4, triton 3.2.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# 2) the consistent inference/RL trio (joint resolve)
pip install vllm==0.8.4 ray==2.46.0 deepspeed==0.16.4
#    CAUTION: this pulled bleeding-2026 transformers 5.10.2 / tokenizers 0.22 / hub 1.17 / numpy 2.2 —
#    exactly the Midway "transformers must be <5" hazard.

# 3) pin ROLL's tested versions + smoke extras (downgrades the above)
pip install transformers==4.51.2 "tokenizers<0.22" "numpy<2.0" \
            datasets==3.1.0 peft==0.12.0 accelerate==0.34.2 \
            tensordict modelscope "tyro>=0.5.7" pydantic loralib einops isort jsonlines \
            deprecated dacite codetiming more_itertools wandb math-verify hydra-core omegaconf \
            gym "gymnasium[toy-text]" gym_sokoban "trl>=0.11,<0.19" \
            open_spiel matplotlib tensorboard
```

Resolved (verified by import): `torch 2.6.0+cu124`, `vllm 0.8.4`, `ray 2.46.0`,
`deepspeed 0.16.4`, `transformers 4.51.2`, `tokenizers 0.21.4`, `numpy 1.26.4`,
`trl 0.18.2`, `open_spiel 1.6.15` (`import pyspiel` → 122 games),
`tensorboard 2.20.0`.

Two **harmless** pip warnings: `cupy-cuda12x 14.1.1` and
`opencv-python-headless 4.13` want `numpy>=2`, but we hold `numpy 1.26.4`. Neither
is imported on the text tic-tac-toe + deepspeed path (Midway confirmed: `grep
"import cv2" roll/` is empty; cupy isn't hit single-node). The full ROLL agentic
import surface loads cleanly under numpy 1.26.4, so the warnings are cosmetic. If
a future path imports them, pin `cupy-cuda12x<14` / `opencv-python-headless<4.10`.

---

## Source patch — `RecvBucketManager` stub (deepspeed-only build)

`roll/distributed/strategy/vllm_strategy.py:14` unconditionally does
`from mcore_adapter.models.converter.convert_utils import RecvBucketManager`, and
that module does `from megatron.core import mpu` at load. With megatron-core not
installed (smoke), importing `VllmStrategy` (the `actor_infer` role) would die
with `ModuleNotFoundError`. This is the exact blocker Midway hit at smoke v5.

Fix (applied to the tree): wrap the import in `try/except ImportError` with a
minimal stub. `VllmStrategy` only ever calls `RecvBucketManager()` and `.clear()`;
`process_bucket()` is the megatron→vllm weight-sync path, unreachable with a
deepspeed actor, so the stub raises `NotImplementedError` there. Verified: with
the stub, `from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline`
imports cleanly.

Patches the venv path does **not** need (that the Midway source-install did):
- **log_monitor `gcs_publisher`** shim — that was a ray ≥2.48 API change; at the
  pinned `ray 2.46.0` ROLL's `log_monitor.py` works unmodified.

---

## The three Polaris files (analogs of the Midway trio)

| File | Role | Key differences from Midway |
|---|---|---|
| `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_smoke.yaml` | hydra config (Qwen2.5-0.5B, deepspeed ZeRO-2, 3 steps, tensorboard) | `pretrain` → eagle model store; vLLM `gpu_memory_utilization` 0.5 → **0.3** (40 GiB A100 vs 140 GiB H200) |
| `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_polaris.sh` | in-job launcher | NO container/`--nv`. Keeps Ray cleanup; adds `set -o pipefail`; the libcuda `LD_PRELOAD` now finds the **host** driver via `ldconfig` (no `/.singularity.d/libs`), kept as harmless insurance |
| `scripts/train_polaris.pbs` | PBS wrap | `#PBS` directives; `-l filesystems=home:eagle`; module-load conda + `CUDA_HOME=cuda-12.4.1` + `CC/CXX=gcc-12`; venv activate; TMPDIR on node-local SSD; caches on eagle; `HF_HUB_OFFLINE=1`; `PYTHONPATH=repo` (no `mcore_adapter/src` — stub covers it) |

**deepspeed JIT build prerequisites (the trickiest wrap detail).** deepspeed
JIT-compiles `fused_adam` on first use via `nvcc` + a host gcc. We set
`CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1` (nvcc 12.4.131, exact match to
torch's cu124) and **override** the conda module's `CC=gcc-14` with
`CC=/usr/bin/gcc-12` (`g++-12`): gcc-12 is ≥9 (deepspeed's floor) and ≤13.2 (CUDA
12.4 nvcc's host-compiler cap). The conda module's gcc-14 would be rejected by
nvcc 12.4. (We do NOT add the toolkit's `lib64` to `LD_LIBRARY_PATH` — torch's
bundled cu124 libs must win at runtime.)

---

## Validation ladder

1. **Toolchain probe (login-node equivalent of the handoff's probe job).**
   Successful. The venv imports the full stack and the ROLL agentic pipeline;
   hydra `compose()` + `from_dict(AgenticConfig, …)` resolves and
   schema-validates the smoke config. (No separate `probe_container_polaris.pbs`
   is needed on the native path — there's no container to test; `nvidia-smi`/GPU
   torch are exercised by the smoke job's own preamble.)
2. **Smoke (deepspeed, Qwen2.5-0.5B, 3 steps).** Successful — job 7186746,
   2026-06-06, after the 10-layer fix stack chronicled in the decisions log.
   See "GREEN — MARSHAL smoke test" below.
3. **Scale / megatron (Qwen3-4B, `megatron_train` TP=4).** In progress
   (2026-06-12) — see the "Megatron scale-up" section below for the toolchain
   extension, placement engineering, and its own validation ladder.

---

## GREEN — MARSHAL smoke test (deepspeed) — Successful, 2026-06-06

**The MARSHAL tictactoe self-play training loop closed end-to-end on ALCF Polaris.**

| Field | Value |
|---|---|
| PBS jid | **7186746** (`debug` queue) |
| Node | `x3105c0s1b1n0` |
| Config | `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_smoke.yaml` (Qwen2.5-0.5B-Instruct, deepspeed ZeRO-2, vLLM V1, **roles on distinct GPUs 0/1/2**, env_groups=2, val disabled) |
| Steps | **3 / 3** — `pipeline step 0/1/2 finished` → `pipeline complete!` |
| Training wallclock | step-0 start 21:39:39 → `pipeline complete!` 21:41:55 = **~2m16s** for 3 steps (total job ~8 min incl. tarball stage + model load) |
| Real metrics (step 2) | `actor/pg_loss`, `actor/kl_loss=0.00144`, `actor/total_loss=0.000288`, `actor_train/grad_norm=1.463`, `critic/ref_log_prob/mean=-0.4618`, `system/tps=242.6`; self-play `env/TicTacToe/winner=1.0` |
| **Checkpoint** | `results/.../7186746_20260606-213348/actor_train-0/checkpoint-2/` (**12 G** full DeepSpeed ckpt: `pytorch_model.bin`, ZeRO optimizer state, `zero_to_fp32.py`, tokenizer) + `pipeline/checkpoint-2/` (heavy blobs pruned 2026-06-12 after verification; `checkpoint_listing_proof.txt` retained in the run dir) |
| **TensorBoard** | `results/.../7186746_20260606-213348/tensorboard/events.out.tfevents.1780781698.x3105c0s1b1n0.654630.0` (49 K) |
| Exit | wrap log: **`Training exited with code: 0`** → `Cleanup complete`; job ended on its own (no qdel) |
| Log | `results/.../7186746_20260606-213348/logs/custom_logs.log` ; wrap `logs/wrap_7186746.log` |

**Artifact that proves the loop closed:** `pipeline complete!` (agentic_pipeline.py:381) after 3
rollout→reference→advantage→DeepSpeed-REINFORCE→model_update cycles, a 12 G `checkpoint-2` on
disk, non-trivial training metrics (grad_norm, kl_loss, tps) in TensorBoard, and the wrap's
**`Training exited with code: 0`**.

> Benign noise: during `ray.shutdown` a RequestScheduler prints `Fatal Python error:
> PyGILState_Release` — it is AFTER `pipeline complete!`, does not affect the exit, and the wrap
> still reports `Training exited with code: 0` and runs cleanup. (Contrast a *rollout* failure,
> which DOES hang the driver — e.g. the val-cascade jid 7186742 sat 45 min until qdel'd.)

### The full fix stack that got to GREEN (each layer was a separate failure — see decisions log)
1. venv tarball → node-local SSD (eagle small-file `import torch` hang)
2. `get_node_ip()` → `ray.util.get_node_ip_address()` (air-gapped compute, no 8.8.8.8)
3. OpenBLAS/OMP thread caps =1 (`pids.max=4096`)
4. `env_groups 16→2` (env-process fan-out)
5. **1 worker/role, roles on distinct GPUs 0/1/2** (fit pids.max + cross-device weight-sync)
6. persistent `TORCH_EXTENSIONS_DIR` + `MAX_JOBS=4` (fused_adam JIT vfork burst)
7. `RAY_NUM_CPUS=16` (shrink Ray idle-worker pool)
8. `roll/third_party/vllm/vllm_0_8_4/llm.py:update_parameter` `.cpu().float()` for V1 (defense)
9. **distinct-GPU placement → NCCL broadcast weight-sync** (THE model_update fix; avoids vLLM-0.8.4 V1 tensor-serialization bug)
10. `agentic_pipeline.py` skip validation when `eval_steps > max_steps` (fewer startup actors + no step-0 val crash)

### Reproduce
```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/MARSHAL
qsub -v MARSHAL_VENV_TARBALL=/lus/eagle/projects/lighthouse-uchicago/members/mehta5/marshal-train-venv.tar \
     scripts/train_polaris.pbs
```
> Residual flakiness: the startup EAGAIN race vs `pids.max=4096` still loses ~half the time at
> RolloutScheduler creation (a lost run hangs — qdel and resubmit). A clean fix would be an ALCF
> ticket to raise the debug-node per-job cgroup `pids.max`.

---

## Decisions / changes log

- **2026-06-04 → 06-05 — Orientation.** Confirmed we're on Polaris
  (`polaris-login-02`, PBS). Found the handoff's container path blocked here (no
  `.sif`, Aliyun registry unreachable, `container_extras/` `.gitignore`'d). Per
  the user, pivoted to a **native venv** built from ROLL's pinned stack. Grounded
  the approach in the ALCF Polaris docs (hardware, running-jobs, python/pytorch/
  deepspeed pages). Corrected the account name to **`lighthouse-uchicago`** via
  `sbank` (the handed-down "Uchicago-lighthouse" is rejected by PBS).
- **2026-06-05 — Built `marshal-train` venv** on eagle (clean `venv` off
  `conda/2025-09-25` base py 3.12.11). Installed torch 2.6.0+cu124, then the
  vllm/ray/deepspeed trio, then pinned ROLL deps to their `requirements_common`
  versions (downgrading the bleeding-edge transformers 5.10.2 → 4.51.2 etc.).
  Verified the full ROLL agentic import surface.
- **2026-06-05 — Patched `vllm_strategy.py`** with the `RecvBucketManager`
  try/except stub (megatron-core absent on the deepspeed-only smoke). Confirmed
  the log_monitor shim is unnecessary at ray 2.46.0.
- **2026-06-05 — Authored the Polaris trio** (`*_polaris_smoke.yaml`,
  `*_polaris.sh`, `train_polaris.pbs`); staged `Qwen2.5-0.5B-Instruct` to the
  eagle model store; hydra dry-run validated.
- **2026-06-05 — Submitted smoke** `qsub scripts/train_polaris.pbs` → first try
  rejected (`Project Uchicago-lighthouse` not found); fixed `-A` to
  `lighthouse-uchicago` → **job 7185571** queued on `debug`.
- **2026-06-05 — Smoke attempt 1 (jid 7185571) HUNG on a bad node.** Queued ~73 min
  (debug contention; comment `Insufficient amount of resource: queue_tags`), started
  04:56 on `x3004c0s25b0n0`, then hung in the early wrap setup: `resources_used.cput
  = 00:00:00`, no `ROLL_OUTPUT_DIR` created, PBS `.OU`/`.ER` empty (still buffered).
  `run_count = 2`. `qdel` returned 0 but could **not** promptly reap it (job stayed
  `R`, walltime kept advancing) — a process wedged in an uninterruptible (D-state)
  syscall, the signature of a hung GPU / `nvidia-smi`. Conclusion: **bad node**, not
  a setup flaw.
  * Hardened `scripts/train_polaris.pbs` so the next run is diagnosable:
    (a) `exec > >(tee logs/wrap_<jid>.log) 2>&1` streams the wrap's stdout/stderr to
        a live eagle file (PBS only flushes `.OU`/`.ER` at job end, hiding early
        hangs — watch with `tail -f logs/wrap_<jid>.log`);
    (b) `timeout 60 nvidia-smi` so a wedged GPU can't hang the whole job;
    (c) `>>> phase:` markers between setup stages to pinpoint where progress stops.
- **2026-06-05 — Smoke attempt 2 (jid 7185610) ALSO hung — at `nvidia-smi`, on a
  DIFFERENT node (`x3101c0s37b1n0`).** The live wrap log proved the env setup is
  fully correct: modules loaded, venv activated (python 3.12.11, nvcc 12.4,
  CC/CXX gcc-12), then hung at the `nvidia-smi` line — `timeout 60` did NOT rescue
  it (unkillable D-state). Ran to walltime: `Exit_status -29`,
  `resources_used.cput 00:00:01`, **`resources_used.ngpus 0`** (never engaged a
  GPU), with a ~38-min prologue gap (R 06:11 → script 06:49). `nvidia-smi` resolves
  correctly to `/usr/bin/nvidia-smi` (no rogue binary on `PATH`). Two different
  nodes both wedging at nvidia-smi with zero GPU engagement ⇒ a **GPU/node-health
  problem on the debug nodes**, not our build.
  * Made the wrap fail-FAST + self-diagnosing so we never burn another full hour:
    (a) `nvidia-smi` now runs fully backgrounded/detached — it can never block;
    (b) a `timeout 180` **torch CUDA probe** aborts in ~3 min (exit 42) if the GPUs
        aren't usable, and reports whether torch sees 4 A100s (the ngpus=0 question);
    (c) line-buffered the live log (`stdbuf -oL tee`).
- **2026-06-05 — Smoke attempt 3 (jid 7185629): GPUs are HEALTHY, but torch CUDA
  init hangs.** Backgrounded `nvidia-smi` returned this time and showed **4 healthy
  A100-SXM4-40GB** (driver 570.124.06 / CUDA 12.8, idle, ECC clean) — so the GPUs
  are fine and our env is correct. But the torch CUDA probe produced no output and
  the fail-fast fired (`Exit_status 42` at 33 min, vs a full-hour hang). The old
  probe was block-buffered (`python -c` without `-u`), so its progress prints were
  lost when `timeout` killed it — couldn't tell which torch call blocked.
- **2026-06-05 — Smoke attempt 4 (jid 7185641): wasted on a probe quoting bug.**
  The inline `python -u -c '...'` had `\"`-escaped quotes inside bash single-quotes
  → `SyntaxError`, so the probe failed instantly (rc=1) without testing torch.
  Fix: moved the probe to **`scripts/polaris_gpu_probe.py`** (granular, unbuffered,
  per-step elapsed timestamps) and the wrap now runs `python -u scripts/polaris_gpu_probe.py`.
  Validated locally first (login node: `import torch` works but takes **47 s** —
  quantifying the eagle/Lustre slowness behind the long startups; `is_available=False`
  → exit 3 as expected).
- **2026-06-05 — Smoke attempt 5 (jid 7185650): `import torch` itself HANGS on the
  compute node.** Clean granular result: `[probe +0.0s] python started`, then NOTHING
  — `import torch` never completed within the 300 s timeout (`rc=124`). On the LOGIN
  node the same import takes **47 s** and completes. So the eagle-resident venv loads
  catastrophically slowly (or hangs) on compute nodes. GPUs are healthy (attempt 3),
  the env is correct. Node `x3101c0s37b1n0` (also attempt 2); with `x3005` (attempt 3,
  also hung in torch), ≥2 nodes hang in torch import/init. ⇒ **Polaris I/O (eagle
  Lustre, poor at the many-small-files access pattern a Python venv hammers) or a
  torch-CUDA-init hang — a system/environment issue, not our build.**
  * **Likely fix to try:** stage the venv to the node-local RAID0 SSD (`/local/scratch`)
    and import from there (the canonical ALCF remedy for "python env on Lustre is slow").
- **2026-06-05 — Root cause CONFIRMED: eagle small-file latency, not the GPUs/torch.**
  Measured on the login node: a single 988 MB model read off eagle = **3 s (422 MB/s)**
  — big sequential reads are fast — but copying the venv's **71,700 files** off eagle
  took **1137 s (~19 min)**. So Lustre is fine for big files and catastrophic for the
  many-small-file metadata storm `import torch` (thousands of `.so`/`.py` opens)
  triggers on a cold compute node. The venv-on-eagle is the whole problem.
  * Checked for a prior working Polaris venv convention to copy (upstream / CMU / the
    `decrypto` port): **none exists** — `decrypto/polaris_handoff_prompt.md` is only a
    handoff prompt (never executed; no `.pbs`, no venv). We're the first real Polaris
    bring-up here, so the venv-location choice is ours.
  * `/home` is writable (45 G quota, fits) but is ALSO Lustre (`/agile/home`) — same
    small-file risk; `/soft` is fast but read-only. Only node-local SSD is both fast
    and writable (but ephemeral, so it needs per-job staging).
- **2026-06-05 — Solution: pack the venv into ONE tarball, stage to node-local SSD.**
  Pre-packed the venv into a single 8.3 G tarball (`marshal-train-venv.tar`, 1165 s
  one-time). The wrap now (restructured): sets up node-local `/local/scratch` early,
  and if `MARSHAL_VENV_TARBALL` is set, reads that one big file (~20 s at 422 MB/s) +
  extracts to local SSD (fast local writes), then runs from there — turning 71.7k slow
  metadata ops into one fast sequential read, per job. Also added a `MARSHAL_VENV`
  override + **manual `pyvenv.cfg`-based activation** (a relocated venv's `activate`
  hardcodes the original path, so it must NOT be sourced). Keeps the proven torch-2.6
  stack — no version juggling.
- **2026-06-05 — Both parallel jobs killed by a BAD NODE, not our code.** jid 7185659
  (home) and jid 7185664 (tarball→SSD) **both landed on `x3016c0s13b1n0`** and both ran
  the full hour with **0-byte output / no wrap log** — the job script never executed
  (stuck PBS prologue). So `x3016c0s13b1n0` is a bad node (broken prologue that eats the
  whole walltime), and the scheduler kept re-assigning it (a broken node sits idle, so
  it's always "available"). **The tarball fix is therefore UNTESTED** — neither job got
  far enough to extract the tarball or run the probe. This is an **ALCF infrastructure
  issue** (debug-node prologue hangs), not ours: the env, the stub, and the tarball
  staging are all sound; we just can't get a healthy node to run them.
  * Recurring symptom across the bring-up: long (~30 min) or stuck (full-hour) PBS
    prologues on debug nodes, and `0-byte .OU` walltime kills (`Exit_status -29`).
  * Resubmitted as **jid 7185695** (node `x3101c0s37b0n0`, a DIFFERENT node) — SAME
    failure: full hour, 0-byte output, script never ran. So it is NOT one bad node.
- **2026-06-05 — DIAGNOSIS: ALCF Polaris filesystem/prologue OUTAGE (not our code).**
  `pbsnodes -l` shows MANY debug nodes offlined right now with filesystem/prologue
  failures: "failed to mount filesystem", "failed mount check", "node offlined due to
  script timeout", "offlined by hook 'prologue_hook' due to hook error" (x3009, x3014,
  x3016 ×5, x3004, x3005, x3101, x3102, x3210, …). The debug queue is mostly Held
  (12 Hld / 3 Run / 5 Que). Our jobs hang in the PBS **prologue's filesystem-mount
  step** on the not-yet-offlined nodes → full-hour walltime kill with no script output.
  This is a Polaris-wide infra problem (Lustre mount / prologue), fully independent of
  our setup.
  * **Action: stop resubmitting (futile until ALCF fixes it).** Wait for recovery /
    file an ALCF ticket, then resubmit the (ready) tarball job:
    `qsub -v MARSHAL_VENV_TARBALL=/lus/eagle/.../mehta5/marshal-train-venv.tar scripts/train_polaris.pbs`.
    Everything on our side is built, staged, and verified — one `qsub` onto a healthy
    node should close the loop (extract tarball → fast import → 3 steps → checkpoint).
- **Observation across all attempts:** consistent **~26–38 min prologue gaps**
  (PBS `R` → wrap script start) on every node, plus slow module/venv loads and a
  47 s torch import — Polaris I/O / node-provisioning is sluggish right now,
  independent of our setup.
- **2026-06-06 — OUTAGE CLEARED; tarball fix PROVEN end-to-end.** `pbsnodes -l` down
  to ~5 offline nodes (all isolated GPU-hardware faults — "Missing GPU", "Failed to
  load GPU during boot" — none filesystem/prologue). Resubmitted jid **7186697**
  (node `x3001c0s13b0n0`): started **immediately** (no prologue hang), and the
  node-local-SSD tarball staging worked exactly as designed:
  * tarball extracted **8.4 G → /local/scratch in 12 s** (vs ~19 min reading the
    venv off eagle),
  * **`import torch` in 1.5 s** (the cold-node import hang — gone),
  * GPU probe **`CUDA COMPUTE OK` on A100-SXM4-40GB**, all 4 GPUs visible, rc=0,
  * pipeline launched → Ray placement group `[[0,1,2,3]]` → tensorboard tracker →
    `max_steps: 3` → wrapped ActorWorker to `ray.remote()`.
  ⇒ The eagle-small-file root cause is **definitively solved**; the tarball approach
  is the right pattern for any Python env on Polaris Lustre.
- **2026-06-06 — Real blocker found & FIXED: ROLL's `get_node_ip()` needs the public
  internet.** jid 7186697 then died in `AgenticPipeline.__init__` → `Cluster._create_workers`
  → every `ActorWorker.__init__` raised:
  ```
  File "roll/distributed/executor/worker.py", line 100, in get_node_ip
      s.connect(("8.8.8.8", 80))
  OSError: [Errno 101] Network is unreachable
  → ray.exceptions.ActorDiedError: actor_train-0:ActorWorker.__init__() failed
  ```
  ROLL resolved each worker's node IP by opening a UDP socket to **`8.8.8.8:80`**
  (Google DNS) and reading the local sockname. **Polaris compute nodes are air-gapped**
  (no route to the public internet) → "Network is unreachable" → all workers die at
  construction → pipeline aborts. (The earlier `SIGABRT`/EAGAIN "Resource temporarily
  unavailable" thread-spawn noise during Ray's worker-pool storm was a RED HERRING —
  Ray recovered from it; this IP call was the actual killer. `ulimit -u` on login is
  already 2 M, so the thread limit is not the bottleneck.)
  * **Fix (`roll/distributed/executor/worker.py`):** resolve the IP via Ray itself —
    `ray.util.get_node_ip_address()` (returns the same `10.201.0.71` Ray already uses,
    no internet needed), falling back to the 8.8.8.8 trick then `gethostbyname(hostname)`
    only if that fails. Grep confirms this was the **only** `8.8.8.8` call site in the repo.
  * Note: `roll/` is imported from the **repo on eagle** (PYTHONPATH = PROJECT_ROOT),
    NOT from the venv tarball — so source patches like this and the `RecvBucketManager`
    stub are picked up on the next `qsub` with **no tarball rebuild**.
  * Resubmitted as jid **7186701** (node `x3006c0s13b0n0`) with the fix in place.
- **2026-06-06 — Next blocker found & FIXED: thread/process explosion (cgroup pids.max),
  not OOM.** jid 7186701 got *much* further with the IP fix — Ray came up clean
  (`GPU: 4.0, CPU: 64`), and ALL 9 workers were constructed (`actor_train-0..3`,
  `actor_infer-0..3`, `reference-0`) — then `reference-0` died:
  `ActorDiedError ... Worker exit type: SYSTEM_ERROR ... connection error code 2`.
  Ray's generic message lists OOM/SIGKILL/SIGSEGV as candidates, but the worker's own
  stdout named the real cause explicitly:
  ```
  (reference-0) OpenBLAS blas_thread_init: pthread_create failed for thread 26 of 64:
      Resource temporarily unavailable
  (reference-0) ... ensure that your address space and process count limits are big enough
  (reference-0) ... or set a smaller OPENBLAS_NUM_THREADS to fit into what you have available
  (reference-0) ... RLIMIT_NPROC 2060880 current, 2060880 max
  ```
  Each of the 9 colocated worker processes spawns **64 OpenBLAS threads** (one per
  hardware thread) + 64 OMP + torch/Ray/CUDA threads. 9× that exceeds the PBS job's
  **cgroup `pids.max`** → `pthread_create` returns EAGAIN → the last worker to start
  (`reference-0`) aborts. **Not OOM** (0.5B model, 512 GiB RAM) and **not** the
  `ulimit -u` nproc limit (`RLIMIT_NPROC` is already ~2 M) — it's the cgroup thread
  cap, which we cannot raise. (The earlier scattered SIGABRTs — `posix_thread::start_thread`
  in `CoreWorker::HandleExit` — were the same EAGAIN hitting *surplus pool* workers on
  exit; non-fatal until it finally hit a real worker.)
  * **Fix (`run_agentic_pipeline_tictactoe_selfplay_polaris.sh`):** cap every process's
    CPU thread pools before launching python (so all Ray workers inherit it before their
    first numpy/OpenBLAS import) — `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=
    NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=RAYON_NUM_THREADS=4`, `TOKENIZERS_PARALLELISM=false`.
    Smoke perf is irrelevant; this keeps total threads well under the cgroup cap.
  * Resubmitted as jid **7186704** with the thread caps in place.
- **2026-06-06 — Thread caps WORKED but exposed the real wall: too many PROCESSES
  (env fan-out) vs cgroup pids.max.** jid 7186704 got the furthest yet — real workers
  survived (`ActorDiedError`=0), `actor_train` reached **DeepSpeed init + JIT-compiling
  `fused_adam`** via `nvcc -ccbin /usr/bin/gcc-12` (our CUDA-12.4/gcc-12 toolchain,
  working). OpenBLAS now tried only "thread 1 of **4**" (cap took effect, down from 64) —
  but **even 4 threads/process still failed** with EAGAIN, and the compile died on:
  ```
  gcc-12: fatal error: cannot execute '.../cc1plus': vfork: Resource temporarily unavailable
  → RayTaskError(ImportError): ActorWorker.initialize() ... fused_adam jit_load failed
  ```
  Process census on the node: **112 `RequestScheduler` + 69 `_QueueActor` + 11 `ActorWorker`
  ≈ 190 Ray processes**. The agentic pipeline spawns one RequestScheduler + _QueueActor
  **per environment instance** = `env_groups * group_size` = (16×4 train)+(16×1 val) = 80
  envs. That many processes' combined threads exhaust the PBS job's **cgroup pids.max**, so
  `pthread_create`/`vfork` fail regardless of per-process thread caps. (`RLIMIT_NPROC` is
  ~2 M — NOT the limit; the cgroup pids cap is, and we can't raise it.) Confirms the earlier
  reasoning: cut the BLAS threads (done) AND cut the process count.
  * **Fix 1 — slash env fan-out (`*_polaris_smoke.yaml`):** `env_groups 16→2` + matching
    `n_groups [16]→[2]` for both train & val (kept `group_size`). 80 envs → 10 → ~35 total
    processes. Verified safe against `agentic_config.py`: the only asserts are
    `max_traj_per_env >= traj_per_env` (auto-satisfied; it defaults to `traj_per_env`), and
    reducing env_groups just raises `traj_per_env` (more sequential trajectories/env). Smoke
    only needs the loop to close, so parallel-env count is irrelevant.
  * **Fix 2 — persist the JIT build (`train_polaris.pbs`):** `TORCH_EXTENSIONS_DIR=$BASE/torch_extensions`
    (compile `fused_adam` once, cache on eagle, every later rank/job loads the `.so` instead
    of re-forking a compiler) + `TORCH_CUDA_ARCH_LIST=8.0` (target only A100/sm_80; also
    silences the in-worker "TORCH_CUDA_ARCH_LIST is not set" warning).
  * Resubmitted as jid **7186707** (node `x3001c0s7b1n0`).
- **2026-06-06 — env_groups reduction did NOT help; STOPPED GUESSING and MEASURED the
  limit.** jid 7186707 (env_groups=2) died the same way — `RolloutScheduler.__init__`
  (`rollout_scheduler.py:42`) → `RequestScheduler`/`val_env-0`: `thread: Resource
  temporarily unavailable` → `ActorDiedError`. Reading `rollout_scheduler.py`: each
  RolloutScheduler creates exactly **one** RequestScheduler actor (2 total, train+val),
  so the "~110 RequestScheduler" earlier was log-LINE count from one chatty actor, not
  processes — env_groups was never the driver. So I submitted a tiny diagnostic
  (`scripts/polaris_limits_probe.pbs`, jid **7186710**, node x3007) that dumps the cgroup
  limits + a live thread-spawn test. **Result (definitive):**
  ```
  /proc/self/cgroup: 0::/jobs/7186710
  /sys/fs/cgroup/jobs/7186710/pids.max   = 4096      # whole-job thread+proc cap
  /sys/fs/cgroup/jobs/7186710/cpu.max    = 6000000 100000   # = 60 cores
  /sys/fs/cgroup/jobs/7186710/memory.max = 515396075520     # 512 GiB (not the issue)
  ulimit -u = 2060880 (RLIMIT_NPROC — red herring), open files (-n) = 16384
  practical test: clone failed after 4093 threads: "can't start new thread"
  ```
  ⇒ **The PBS job cgroup hard-caps the entire job at `pids.max=4096` threads/processes.**
  It is root-owned (parent `/jobs` is `pids.max=max`); we cannot raise it from inside the
  job. Every vLLM/DeepSpeed/CUDA/NCCL/gRPC worker spawns dozens–hundreds of threads that
  `OMP_NUM_THREADS` does NOT govern, so 9 GPU workers (4 train + 4 infer + 1 reference at
  4 GPUs) cross 4096 — especially during the fused_adam compile burst. This single fact
  explains every EAGAIN we chased (OpenBLAS-64, OpenBLAS-4, RequestScheduler, reference-0,
  the compile vfork): all were the 4096 ceiling hit at different thread counts.
  * **Fix — fit under 4096 by shrinking GPU fan-out (`*_polaris_smoke.yaml`):**
    `num_gpus_per_node 4 → 1`. Roles become 3 colocated workers (actor_train + actor_infer
    + reference all on GPU 0 — ROLL's colocate design supports this), ~1/4 the threads, well
    under the cap. 0.5B model fits trivially (vLLM gpu_mem_util=0.3). CLAUDE.md sanctions
    "smallest model on 1–2 GPUs" for the smoke.
  * Kept: thread caps (OMP=4…), env_groups=2, persistent TORCH_EXTENSIONS_DIR, IP fix, stub.
  * Resubmitted as jid **7186711** (node `x3001c0s19b1n0`).
  * NOTE for scale-up: the 4-GPU / megatron path needs the per-job thread budget kept under
    4096 — fewer threads/process and/or an ALCF ticket to raise the job cgroup `pids.max`.
- **2026-06-06 — 1 GPU cleared ALL infra blockers; reached the training loop. Next (last)
  blocker: vLLM V1 weight-sync.** jid 7186711 (1 GPU) sailed through everything: fused_adam
  JIT-compiled & cached to `$TORCH_EXTENSIONS_DIR` on eagle, DeepSpeed ZeRO-2 engine init,
  vLLM loaded the model (`GPU KV cache size: 721,696 tokens`) — then died at the first
  **model_update** (push DeepSpeed actor weights → vLLM engine):
  ```
  base_pipeline.py:68 model_update → model_update_group.py:152
  deepspeed_strategy.py:397 model_update → vllm_strategy.py:365 update_parameter
  third_party/vllm/vllm_0_8_4/llm.py:210  self.collective_rpc("update_parameter", args=(... CUDA weight ...))
  vllm/v1/engine/core_client.py _send_input
  TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() ...
  ```
  Root cause: ROLL's per-parameter `update_parameter` (llm.py:209) has **no V1 branch** — it
  passes the CUDA weight straight to `collective_rpc`. Under the vLLM **V1** engine the engine
  core runs in a separate process and msgpack-serializes RPC args ⇒ GPU tensor can't serialize.
  Only the *bucket* path (`update_parameter_in_bucket`, llm.py:213) does `.cpu().tolist()` for
  V1; the path the DeepSpeed strategy uses does not. So ROLL's weight-sync requires the **V0**
  engine (in-process workers, CUDA tensors broadcast via NCCL). **Midway never set VLLM_USE_V1**
  (grep: absent everywhere) — its vLLM defaulted to V0, so the bug never fired; ours defaults
  to V1.
  * **Fix (`run_..._polaris.sh`):** `export VLLM_USE_V1=0` (before vLLM import; inherited by the
    Ray actor_infer worker). Also added `MAX_JOBS=4` (cap ninja's compile-burst for any future
    JIT) + `TORCH_CUDA_ARCH_LIST` default.
  * Bonus: 7186711 left a built `fused_adam.so` in the persistent `$TORCH_EXTENSIONS_DIR`, so
    every later run LOADS it (no compile burst at all).
  * Resubmitted as jid **7186716** (node `x3001c0s7b0n0`) — V0 engine + cached fused_adam.
- **2026-06-06 — pids.max=4096 is a TIGHT fit even at 1 GPU; added real headroom.** jid 7186716
  (1 GPU + V0 + cached fused_adam) died EARLIER than 7186711 — at `RolloutScheduler.__init__`
  (`rollout_scheduler.py:42`), its RequestScheduler/EnvironmentWorker hitting the same EAGAIN.
  Same config as 7186711 (which reached model_update) ⇒ **success was luck**: the startup
  "thundering herd" (~17 processes importing torch/vLLM/scipy and spawning threads at once)
  peaks near 4096 and intermittently kills a critical actor. Needed margin, not luck.
  * Diagnosis of the biggest thread source: **Ray prestarts ~one idle worker per detected CPU
    (~64)**, each carrying ~30+ Ray threads (the contiguous-pid SIGABRT blocks we saw on exit).
    That idle pool alone is ~2000 threads.
  * **Fix (`run_..._polaris.sh`):** `RAY_NUM_CPUS=16` — ROLL's `start_ray_cluster()` (initialize.py:45)
    forwards it to `ray start --num-cpus`, shrinking the idle pool ~4× (ample for our ~10 CPU
    actors). Plus dropped the BLAS/OMP caps **4 → 1** (`OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/RAYON=1`)
    for max per-process thread reduction — smoke needs no CPU math throughput.
  * Resubmitted as jid **7186717** (node `x3001c0s7b1n0`) — 1 GPU + V0 + cached fused_adam +
    RAY_NUM_CPUS=16 + single-thread BLAS.
- **2026-06-06 — V0 was the wrong fix for the TypeError: it's THREAD-HEAVIER than V1. Keep V1,
  patch the weight-sync instead.** jid 7186717 confirmed `RAY_NUM_CPUS=16` applied (`ray start
  ... --num-cpus=16`, `'CPU': 16.0`) and OMP=1 — yet STILL died at `RolloutScheduler.__init__`
  (EAGAIN), *earlier* than the V1 run 7186711 (which reached model_update). The discriminating
  variable is the engine: **vLLM V0 spawns markedly more startup threads than V1**, so under
  pids.max=4096 V0 dies before training while V1 reaches the loop. So forcing V0 (to dodge the
  model_update TypeError) was counterproductive.
  * **Correct fix = keep V1 + fix the one V1-incompatible call.** DeepSpeed `model_update` has
    two transfer paths: NCCL `collective.broadcast` (GPU tensor, serialization-free, V1-safe)
    and a **P2P** path `update_parameter.remote(weight=<cuda tensor>)`. In our 1-GPU *colocated*
    setup (train & infer both on GPU 0) it takes the P2P path, and ROLL's
    `vllm_0_8_4/llm.py:update_parameter` passed the CUDA tensor straight into `collective_rpc`
    → V1 cross-process msgpack can't serialize a GPU tensor → the TypeError. Receiver
    (`worker_helper.py:update_parameter` → vLLM `load_weights`) copies a CPU-source tensor into
    the GPU param fine, so the fix is one-sided.
  * **Patch (`roll/third_party/vllm/vllm_0_8_4/llm.py`):** in `update_parameter`, when
    `envs.VLLM_USE_V1`, `.cpu()` any tensor in args/kwargs before `collective_rpc` (mirrors the
    `.cpu()` the bucket path already does). Reverted launcher to `VLLM_USE_V1=1`.
  * Kept all headroom knobs (1 GPU, RAY_NUM_CPUS=16, OMP=1, env_groups=2, cached fused_adam).
  * Resubmitted as jid **7186720**.
- **2026-06-06 — `.cpu()` advanced the error (cuda → bf16); up-cast to float32 to finish it.**
  jid 7186720 (V1 + .cpu() patch) cleared worker creation (the single RequestScheduler EAGAIN
  was non-fatal — Ray recovered), reached DeepSpeed ZeRO-2 init + vLLM load + KV cache, and the
  `.cpu()` patch turned the model_update error from "can't convert cuda:0 ... to numpy" into a
  NEW one: `TypeError: Got unsupported ScalarType BFloat16`. Cause: vLLM V1's serializer uses
  `tensor.numpy()`, and **numpy has no bfloat16** (the exact reason the bucket path avoids numpy).
  * **Patch refinement (`vllm_0_8_4/llm.py:update_parameter`):** also up-cast bf16/fp16 → float32
    on the host before `collective_rpc` (lossless for bf16; float32 represents every bf16 value).
    The receiver's `load_weights` copies it back into the bf16 GPU param, casting. The advancing
    error chain (cuda→bf16→[expected: pass]) confirms the mechanism is correct.
  * Resubmitted as jid **7186727** — V1 + full bf16-safe weight-sync patch.
- **2026-06-06 — The per-parameter V1 serialization is unwinnable; route weight-sync through
  NCCL broadcast instead (place roles on DISTINCT GPUs).** After winning the startup race
  (jid 7186732), the `.cpu().float()` patch cleared cuda+bf16 — but a THIRD layer appeared:
  `qwen2.py:409 load_weights: assert loaded_weight.shape... AttributeError: 'list' object has
  no attribute 'shape'`. vLLM 0.8.4's `serial_utils.MsgpackEncoder.enc_hook` does
  `self._encode_ndarray(obj.numpy())` for a tensor, but the **decoder only reconstructs a
  tensor when the RPC arg is type-hinted** (`dec_hook(t, obj)`); ROLL's generic `collective_rpc`
  args aren't, so the receiver gets a raw list. This is the "bug in encoder/decoder of vllm
  084" the bucket-path comment names — the per-parameter path is simply not V1-serializable.
  * **Real fix: avoid serialization entirely.** `model_update_group.make_comm_plan()` (line 85)
    chooses **P2P** (`update_parameter.remote(weight=tensor)`, serialized) when src(train) and
    tgt(infer) share `(node, gpu)`, and **NCCL broadcast** (`collective.broadcast`, no msgpack)
    when they're on different GPUs. Our 1-GPU colocation forced P2P. Putting **actor_train=GPU0,
    actor_infer=GPU1, reference=GPU2** (`device_mapping: [0]/[1]/[2]`, `num_gpus_per_node: 4`)
    routes weight-sync over NCCL broadcast — V1-safe, no tensor ever serialized. Still only 3
    GPU worker processes (same pids footprint as colocated-1-GPU), just spread across 3 of the
    4 A100s. The `.cpu().float()` llm.py patch stays as defense for any residual P2P.
  * Startup pids race (~50%, env/scheduler actors vs pids.max=4096) is orthogonal and remains —
    retry past it; the cross-device weight-sync then lets model_update succeed.
  * Resubmitted as jid **7186739** — roles on distinct GPUs (NCCL-broadcast weight-sync).
- **2026-06-06 — `device_mapping` must be a STRING (ROLL eval()s it).** jid 7186739 died at config
  parse: `TypeError: eval() arg 1 must be a string` — ROLL `eval()`s `device_mapping`, so a YAML
  list `[0]` breaks it. Fix: quote them — `device_mapping: "[0]"` / `"[1]"` / `"[2]"`. (jid 7186742.)
- **2026-06-06 — BREAKTHROUGH: NCCL-broadcast weight-sync WORKS; model_update fully cleared.**
  jid 7186742 (distinct GPUs) won the startup race and ran:
  ```
  weight update progress: 100%|██████████| 290/290     <- all params synced, NCCL broadcast, NO serialization
  model_update_end_onload / model_update_end_offload   <- model_update COMPLETED
  val rollout progress(trajectory): 0/16               <- entered the rollout phase; vLLM generating
  ```
  The distinct-GPU placement routed weight-sync through `collective.broadcast` and the V1
  serialization problem is GONE. It then died, but only as a **cascade**: the *validation*
  RolloutScheduler's RequestScheduler had been killed earlier by the startup EAGAIN race, and the
  step-0 eval's `get_batch` used that dead actor. So the lone remaining blocker is the startup
  pids race — here it happened to hit the val scheduler.
- **2026-06-06 — Disable validation for the smoke (fewer startup actors + removes a failure path).**
  The loop evals at `global_step % eval_steps == 0`, and `0 % 100 == 0` fires an implicit step-0
  eval; the val RolloutScheduler is also created unconditionally. With `eval_steps=100 > max_steps=3`
  validation is meaningless for bring-up. **Patch (`agentic_pipeline.py`):** create `val_rollout_scheduler`
  only when `eval_steps <= max_steps` (else `None`), and guard the eval block on it. This drops ~3
  startup actors (better odds vs pids.max=4096) and removes the step-0 val crash. The training
  rollout (which DOES run every step) + model_update (proven) should now reach the 3 steps + checkpoint.
  * Resubmitted as jid **7186746** — distinct GPUs + val disabled.

---

# Megatron scale-up (Qwen3-4B, `megatron_train` TP=4) — started 2026-06-12

Goal: reproduce on Polaris the configuration Midway proved GREEN (jid 50261211):
Qwen3-4B, `actor_train: megatron_train` with TP=4 + sequence_parallel +
distributed optimizer + recompute=full, vLLM rollouts, 3 steps — then a 20-step
proof run (the analog of Midway's GREEN scale-up, jid 50259767). Authored per
`polaris_megatron_handoff_prompt.md`.

## Cluster facts — additions and re-verifications (2026-06-12)

| Item | Value | Evidence |
|---|---|---|
| Queue `debug` | 1–2 nodes, walltime 5 min–1 h, **1 running job/user** | `qstat -Qf debug`, 2026-06-12 |
| Queue `debug-scaling` | 1–10 nodes, walltime 5 min–**1 h** (same cap as debug, just wider) | `qstat -Qf debug-scaling`, 2026-06-12 |
| Queue `preemptable` | 1–10 nodes, walltime up to **72 h**, `max_run=[p:PBS_GENERIC=10]`, preemptible (`-r y`) | `qstat -Qf preemptable`, 2026-06-12 |
| **Login-node per-user cgroup** | `memory.max = 8 GiB`, `pids.max = 256` (`/sys/fs/cgroup/users/<user>/`) | discovered 2026-06-12 when a MAX_JOBS=16 build was OOM-killed; governs all login-node builds |
| cuDNN | modules `cudnn/8.9.7 … 9.13.0` exist under `/soft/modulefiles`; **not needed** — the venv's `nvidia-cudnn-cu12 9.1.0.70` (torch dependency) ships headers + libs | `module avail cudnn`; `site-packages/nvidia/cudnn/{include,lib}` |
| Qwen3-4B geometry | 32 attention heads, **8 KV heads**, 36 layers, hidden 2560, head_dim 128 → TP=4 and TP=2 both divide; KV cache ~144 KB/token | `$BASE/models/Qwen3-4B/config.json` (read 2026-06-12) |
| eagle free space | ~1.1 TB free (90% used) at scale-up start | `df -h /lus/eagle`, 2026-06-12 |
| torch C++ ABI | `torch._C._GLIBCXX_USE_CXX11_ABI = False` → flash-attn wheel must be `cxx11abiFALSE` | venv python, 2026-06-12 |

## Toolchain extension — megatron stack into `marshal-train` (login node, 2026-06-12)

Version targets are the **container-verified set** from `midway_notes.md`
(megatron.core 0.12.3, mcore_adapter 0.6.0.dev0, transformer_engine 2.2.0,
flash_attn 2.7.2, apex@25.04). The Dockerfile's `megatron-core==0.11.0` pin is
a known discrepancy; `mcore_adapter/requirements.txt` pins
`megatron-core>=0.12.0,<0.13.0`, so 0.12.3 is the consistent choice.

Steps actually run (each followed by a pin-guard re-verification of
torch 2.6.0+cu124 / vllm 0.8.4 / ray 2.46.0 / deepspeed 0.16.4 /
transformers 4.51.2 / tokenizers 0.21.4 / numpy 1.26.4):

- **flash-attn 2.7.2.post1** — prebuilt wheel
  `flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl`
  from the GitHub release (cp312 variant of the Dockerfile's cp310 wheel; ABI
  flag matched against the venv's `False`). Installed `--no-deps`. Successful.
- **megatron-core 0.12.3** — first attempt (`pip install megatron-core==0.12.3`)
  was **Unsuccessful and destructive**: pip's resolver dragged in torch 2.12.0,
  numpy 2.4.6, triton 3.7, setuptools 81 and a set of cu13 NVIDIA packages.
  Recovery had a second-order trap: the cu13 packages install into the *same*
  `site-packages/nvidia/<lib>/` paths as torch's cu12 dependencies, so
  uninstalling them deleted cu12 files and broke `import torch`
  (`libcudnn.so.9: cannot open shared object file`). Resolution:
  `pip install --force-reinstall torch==2.6.0 torchvision==0.21.0
  torchaudio==2.6.0 "numpy<2.0"` to relay the clobbered cu12 payloads, then
  `pip install --no-deps megatron-core==0.12.3` (its real deps — torch, numpy,
  packaging — were already satisfied). Regression check (full ROLL agentic
  import surface + flash_attn) passed afterwards. Lesson: **install
  megatron-stack packages `--no-deps` where the dep tree is already pinned.**
- **transformer-engine 2.2.0** — `pip install -v --no-build-isolation
  "transformer-engine[pytorch]==2.2.0"` with `CUDA_HOME=cuda-12.4.1`,
  `CC=gcc-12`, `CUDNN_PATH=$VENV/.../nvidia/cudnn`, `CUDAARCHS=80`.
  `transformer_engine` + `transformer_engine_cu12` arrive as prebuilt manylinux
  wheels; only `transformer_engine_torch` (26 C++ files, no nvcc) compiles
  locally. First attempt at `MAX_JOBS=16` was **Unsuccessful** — cc1plus
  OOM-killed by the login node's 8 GiB user cgroup (`g++-12: fatal error:
  Killed signal terminated program cc1plus`). Retried at `MAX_JOBS=2`:
  Successful (~25 min; import-time findings in the decisions log).
- **apex (NVIDIA, tag 25.04)** — mirror of the Dockerfile invocation: source
  tree cloned to `$BASE/tmp/apex-25.04`, then `pip install -v --no-cache-dir
  --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext
  --parallel 2" $BASE/tmp/apex-25.04` with `MAX_JOBS=2` (login-node 8 GiB user
  cgroup again), `TORCH_CUDA_ARCH_LIST=8.0` (sm_80 only), `CC=gcc-12`,
  nvcc 12.4.131. Compiled ~25 min; produced the full `--cuda_ext` surface
  (`amp_C`, `fused_layer_norm_cuda`, the four megatron fused-softmax
  extensions, `fused_rotary_positional_embedding`,
  `fused_weight_gradient_mlp_cuda`, `syncbn`, `mlp_cuda`, ...). All CUDA
  extensions import cleanly on the login node (linkage proof; first execution
  is rung 2). Build log: `$BASE/tmp/apex_build.log`. Successful.
- **mcore_adapter 0.6.0.dev0** — `pip install --no-deps --no-build-isolation
  ./mcore_adapter` (its three declared deps were already satisfied:
  `megatron-core>=0.12,<0.13` by 0.12.3, `transformers>=4.48` by 4.51.2,
  `accelerate>=0.27.2` by 0.34.2). Two follow-on findings:
  * **`pkg_resources` regression.** `mcore_adapter/.../convert_utils.py:9`
    does `from pkg_resources import packaging`; the megatron-core resolver
    incident (above) had left `setuptools 82.0.1`, which no longer ships
    `pkg_resources` (removed in setuptools 81) → `import mcore_adapter` died
    with `ModuleNotFoundError: pkg_resources`. Resolution:
    `pip install setuptools==75.8.2` (last-era setuptools that ships
    pkg_resources; matches what the Midway container ran). Chose the
    setuptools downgrade over patching the vendored mcore_adapter source so
    the package stays byte-identical to the Midway-proven one.
  * **megatron-core's missing declared deps**, surfaced by `pip check` after
    the `--no-deps` install: installed `zarr 2.18.7` (+ numcodecs 0.15.1) and
    `tensorstore 0.1.84` (+ ml_dtypes 0.5.4) — these back
    `megatron.core.dist_checkpointing`, which mcore_adapter's
    trainer/checkpointing actually use (optimizer state goes through
    `dist_checkpointing.save`), and compute nodes are air-gapped so a lazy
    import miss there would cost a debug job. Deliberately did NOT install
    the remaining declared deps (pytest / pytest-cov / pytest-random-order,
    flask-restful, nltk, nvidia-modelopt): grep of the installed
    `megatron/core` shows zero import sites for nltk/flask/pytest, and
    modelopt only under `inference/modelopt_support/` + `post_training/` —
    none on the training path. numpy stayed 1.26.4 throughout. Successful.

## Placement engineering — memory × pids (the core scale-up problem)

Two hard constraints. (1) **Memory:** Qwen3-4B ≈ 4.0 B params (computed from
config.json geometry: 36 layers × ~101 M + 389 M tied embedding). Megatron
training state at the Midway config's settings (bf16 weights, fp32 grad
accumulation, fp32 master weights + Adam moments) ≈ 18 bytes/param ≈ **72 GB**,
sharded over the TP ranks. (2) **pids:** the job cgroup caps the whole job at
**`pids.max=4096`** (measured, jid 7186710); measured data points — 3 GPU
workers passed (with a residual ~50% startup race), 9–12 GPU workers failed
every time.

| Layout | actor_train | actor_infer | reference | GPU workers | Train state/GPU | Assessment (pre-run) |
|---|---|---|---|---|---|---|
| A. Midway mirror | TP=4 `"[0,1,2,3]"` | `"[0,1,2,3]"` | `"[0,1,2,3]"` | 12 | ~18 GB | Rejected: 12 workers is the measured-fail pids shape |
| **B. TP=4 + single-GPU infer/ref** | TP=4 `"[0,1,2,3]"` | `"[0]"` | `"[1]"` | 6 | ~18 GB | **Primary candidate.** Memory: GPU0 worst case ≈ 18 (train shard) + 14 (vLLM at util 0.35) + contexts ≈ 34 GB < 40 even with no offload; GPU1 ≈ 18 + 8 (ref) ≈ 27 GB. pids: 6 workers sits between the 3-pass and 9/12-fail data points — to be measured (census in the wrap) |
| C. TP=2 distinct GPUs | TP=2 `"[0,1]"` | `"[2]"` | `"[3]"` | 4 | ~36 GB | Fallback only: ~36 GB/rank leaves no activation headroom on a 40 GB card; would need heavy memory dials |

Supporting analysis (code-verified 2026-06-12, before any run):

- **Weight-sync path.** `roll/distributed/executor/model_update_group.py:make_comm_plan`
  assigns each tgt device a src rank, *skipping* a src rank whose (node, gpu)
  collides with the tgt; P2P is only used when there is a single src rank (the
  smoke's 1-GPU case). With 4 distinct TP src ranks, layout B gets a pure
  **NCCL-broadcast bucket path**: `megatron_strategy.model_update` all-gathers
  HF-format buckets (256 MB) across TP and `collective.broadcast`s them;
  under vLLM V1 only bucket *metadata* is msgpack'd
  (`SendBucketManager.meta_to_dict`), never the tensor. The receiving side
  (`roll/third_party/vllm/worker_helper.py`) uses ROLL's own
  `roll.utils.send_recv_utils.RecvBucketManager` — megatron-free. The
  mcore_adapter `RecvBucketManager` import in `vllm_strategy.py` is
  driver-side; with the real package installed the stub's `try` path imports
  the real module and the stub is dormant (kept for the deepspeed-only smoke).
- **vLLM memory dial.** Qwen3-4B KV ≈ 144 KB/token (36 layers × 2 × 8 KV-heads
  × 128 × 2 B). `gpu_memory_utilization: 0.35` → 14 GB = 8 GB bf16 weights +
  ~5.7 GB KV ≈ 40 k tokens — ample for 16 tictactoe trajectories at
  `max_new_tokens: 1024`. (The Midway 0.6 was sized for 140 GB H200s.)
- **vLLM offload.** ROLL's `VllmStrategy.offload_states` calls vLLM 0.8.4
  sleep mode (`enable_sleep_mode=True`, default `sleep_level=1` — KV cache
  freed, weights kept). Layout B's worst case above assumes **no** offload and
  still fits; offload is upside, not a dependency.
- **`use_distributed_optimizer` with DP=1**: there are no extra DP ranks to
  shard optimizer state over, so its savings are likely nil here — the 18
  bytes/param estimate deliberately does not credit it.

## ALCF ticket — raise the per-job cgroup `pids.max` (DRAFT, to be filed)

The durable fix for the whole pids class (including the smoke's residual ~50%
startup race). This session cannot send email/tickets; **the user should file
this with ALCF support (support@alcf.anl.gov / the ALCF help desk portal)**:

> **Subject:** Request: raise per-job cgroup pids.max on Polaris GPU nodes
> (currently 4096)
>
> **Project:** lighthouse-uchicago (allocation 12374). **User:** rmehta1987.
>
> On Polaris compute nodes, the PBS job cgroup (`/sys/fs/cgroup/jobs/<jobid>`)
> caps the entire job at `pids.max=4096` threads+processes. We measured this
> directly (job **7186710**, probe script `scripts/polaris_limits_probe.pbs`
> in our repo): `pids.max = 4096`, and a live clone test failed after 4093
> threads ("can't start new thread"). The parent cgroup `/jobs` is
> `pids.max=max` and the setting is root-owned, so it cannot be raised from
> inside a job.
>
> Our workload is a multi-role RL training pipeline (Ray + vLLM + DeepSpeed /
> Megatron-Core: one training actor group, one vLLM rollout engine, one
> reference-model worker, plus environment actors). Each GPU worker process
> legitimately spawns hundreds of threads (NCCL, gRPC, CUDA, ray core) that
> per-process knobs like OMP_NUM_THREADS do not govern. At 4096 we can only
> run severely shrunken layouts (3–6 GPU workers), and even those lose ~50% of
> submissions to an EAGAIN race during the startup import storm (e.g. jobs
> 7186716/7186717/7186727 died at actor construction with "Resource
> temporarily unavailable" despite OMP/OPENBLAS=1, RAY_NUM_CPUS=16).
>
> Request: raise the per-job `pids.max` on GPU nodes (e.g. to 16384, or make
> it scale with ncpus), or advise on a supported mechanism to request a higher
> limit per job.

Status: drafted 2026-06-12, not yet filed (flagged to the user). Layout
engineering proceeds in parallel per the plan above.

## Scale-up validation ladder

1. **[login] Toolchain extension proven offline** — megatron-stack imports +
   proven-stack regression + hydra compose of the new config; new tarball
   packed under a new name; old tarball intact. Status: **Successful**
   (2026-06-12; pin guard 12/12, stub dormant, compose dry-run clean,
   `marshal-train-megatron-venv.tar` packed alongside the untouched GREEN
   fallback — details in the decisions log).
2. **[debug] On-GPU toolchain probe** — new tarball staged; megatron stack
   imports on-node; minimal flash-attn forward on the A100. Status:
   **Successful** (job 7197416, node `x3004c0s1b0n0`, Exit_status 0: 10.25 G
   tarball extracted in 16 s; torch + megatron.core + TE(+pytorch) +
   flash_attn + apex(+CUDA exts) + mcore_adapter all imported in 14.5 s;
   `flash_attn_func` bf16 forward and a TE LayerNorm executed on the A100;
   stub-dormancy asserted on-node; probe-job pids.peak 83. Bonus finding:
   `pids.peak` exists in the job cgroup on these nodes, so the census records
   true peaks. Log: `logs/wrap_7197416.log`).
3. **[debug] Megatron 3-step smoke** at layout B. Status: **Successful**
   (job 7197427, 2026-06-12, after 3 prior attempts: two startup-race losses
   and one vLLM `max_model_len` fix — see the decisions log and ledger).
4. **[debug or preemptable] 20-step proof run.** Status: **Successful**
   (job 7197442, 2026-06-12, `debug`: walltime math from the 3-step smoke
   predicted ~47 min worst case vs the 60 min cap — measured 37m37s. 20/20
   steps, two incremental megatron checkpoints, pids.peak 2313 with the
   RequestScheduler pool patch. Preemptable was held in reserve and not
   needed.)

## Job ledger — every Polaris log file, its job, and its outcome

`logs/` also contains `pull_*`/`train_midway_*` files; those belong to
`midway_notes.md`'s ledger. PBS spool files are abbreviated `<jid>.OU/.ER`
(full names `logs/<jid>.polaris-pbs-01.….OU/.ER`). Jobs 7185571, 7185659,
7185664, 7185695 have no `wrap_*.log`: 7185571 predates the live tee; for the
other three the job script never ran (PBS prologue hang), which is itself the
finding. `nvidia-smi_<jid>.txt` exists for every job from 7185629 on that
reached the wrap's probe phase.

| Log file(s) | Job id | Queue | What ran | Outcome | Root cause / note |
|---|---|---|---|---|---|
| `logs/7185571.*.OU/.ER` (no wrap log — predates the live tee) | 7185571 | debug | smoke attempt 1 (0.5B deepspeed) | Unsuccessful | Bad node `x3004c0s25b0n0`: wedged in early wrap setup, zero cput, D-state, unkillable via qdel; burned walltime |
| `logs/wrap_7185610.log`, `logs/7185610.*.OU/.ER` | 7185610 | debug | smoke attempt 2 + hardened wrap (live tee, phase markers) | Unsuccessful | Different node `x3101c0s37b1n0` hung at `nvidia-smi` (unkillable D-state, `timeout` did not rescue); `resources_used.ngpus 0`; walltime kill `Exit_status -29` |
| `logs/wrap_7185629.log`, `logs/nvidia-smi_7185629.txt`, `logs/7185629.*.OU/.ER` | 7185629 | debug | smoke attempt 3 (backgrounded nvidia-smi + torch fail-fast probe) | Unsuccessful | GPUs proven healthy (4 idle A100, ECC clean) but the torch CUDA probe produced no output and fail-fast fired (exit 42 at 33 min); probe was block-buffered so the blocking call was unidentifiable |
| `logs/wrap_7185641.log`, `logs/nvidia-smi_7185641.txt`, `logs/7185641.*.OU/.ER` | 7185641 | debug | smoke attempt 4 | Unsuccessful | Wasted run: inline probe had a bash-quoting SyntaxError (rc=1 instantly). Fix became `scripts/polaris_gpu_probe.py` (granular, unbuffered) |
| `logs/wrap_7185650.log`, `logs/nvidia-smi_7185650.txt`, `logs/7185650.*.OU/.ER` | 7185650 | debug | smoke attempt 5 (granular probe) | Unsuccessful | Definitive isolation: `import torch` itself hung >300 s on the compute node (47 s on login). Root cause measured later: eagle small-file latency (71,700 venv files = 1137 s vs one 988 MB file = 3 s) |
| `logs/7185659.*.OU/.ER` (no wrap log) | 7185659 | debug | venv-on-/home comparison run | Unsuccessful | Job script never executed — PBS prologue hang on bad node `x3016c0s13b1n0`; full hour, 0-byte output |
| `logs/7185664.*.OU/.ER` (no wrap log) | 7185664 | debug | first tarball→SSD staging run | Unsuccessful | Same bad node `x3016c0s13b1n0`, same prologue hang; the tarball fix went untested |
| `logs/7185695.*.OU/.ER` (no wrap log) | 7185695 | debug | tarball→SSD retry | Unsuccessful | Different node `x3101c0s37b0n0`, same script-never-ran signature → diagnosed cluster-wide prologue/Lustre outage (`pbsnodes -l`: many nodes offlined on mount failures). Stopped resubmitting until it cleared |
| `logs/wrap_7186697.log`, `logs/nvidia-smi_7186697.txt`, `logs/7186697.*.OU/.ER` | 7186697 | debug | tarball→SSD staging, first run after outage | Unsuccessful overall, but proved the staging fix (8.4 G extracted in 12 s; `import torch` 1.5 s; `CUDA COMPUTE OK`) | Died in worker construction: ROLL `get_node_ip()` dials `8.8.8.8:80`; Polaris compute is air-gapped → `OSError: Network is unreachable`. Fixed via `ray.util.get_node_ip_address()` in `worker.py` |
| `logs/wrap_7186701.log`, `logs/nvidia-smi_7186701.txt`, `logs/7186701.*.OU/.ER` | 7186701 | debug | smoke + IP fix | Unsuccessful | All 9 workers constructed; `reference-0` died: OpenBLAS `pthread_create failed ... Resource temporarily unavailable` — 9 procs × 64 BLAS + 64 OMP threads vs the job cgroup thread cap. Fix: thread caps exported before python |
| `logs/wrap_7186704.log`, `logs/nvidia-smi_7186704.txt`, `logs/7186704.*.OU/.ER` | 7186704 | debug | smoke + OMP/BLAS=4 caps | Unsuccessful | Caps took effect but EAGAIN persisted; fused_adam JIT died (`cc1plus: vfork: Resource temporarily unavailable`); ~190 Ray procs from env fan-out (env_groups=16). Fixes: env_groups→2, persistent TORCH_EXTENSIONS_DIR |
| `logs/wrap_7186707.log`, `logs/nvidia-smi_7186707.txt`, `logs/7186707.*.OU/.ER` | 7186707 | debug | smoke + env_groups=2 | Unsuccessful | Same EAGAIN at `RolloutScheduler.__init__` → stopped guessing, built the limits probe |
| `logs/limits_probe.out` | 7186710 | debug | cgroup limits probe (`scripts/polaris_limits_probe.pbs`) | Successful | Measured the governing constraint: **`pids.max=4096`** for the whole job (cpu.max=60 cores, memory.max=512 GiB); live clone test failed at 4093 threads. RLIMIT_NPROC (~2 M) confirmed a red herring |
| `logs/wrap_7186711.log`, `logs/nvidia-smi_7186711.txt`, `logs/7186711.*.OU/.ER` | 7186711 | debug | smoke at 1 GPU (3 roles colocated) | Unsuccessful | Cleared ALL infra (fused_adam built+cached, ZeRO-2 init, vLLM loaded); died at first `model_update`: vLLM V1 msgpack cannot serialize a CUDA tensor (per-parameter P2P path) |
| `logs/wrap_7186716.log`, `logs/nvidia-smi_7186716.txt`, `logs/7186716.*.OU/.ER` | 7186716 | debug | 1 GPU + `VLLM_USE_V1=0` | Unsuccessful | Died EARLIER than the V1 run (RolloutScheduler EAGAIN): V0 spawns more startup threads than V1 — forcing V0 was counterproductive |
| `logs/wrap_7186717.log`, `logs/nvidia-smi_7186717.txt`, `logs/7186717.*.OU/.ER` | 7186717 | debug | 1 GPU + V0 + RAY_NUM_CPUS=16 + BLAS=1 | Unsuccessful | Knobs verified applied (`--num-cpus=16`, OMP=1) yet still startup EAGAIN under V0 → keep V1, fix serialization instead |
| `logs/wrap_7186720.log`, `logs/nvidia-smi_7186720.txt`, `logs/7186720.*.OU/.ER` | 7186720 | debug | 1 GPU + V1 + `.cpu()` patch in `llm.py:update_parameter` | Unsuccessful | Error advanced: "can't convert cuda:0 tensor" → "Got unsupported ScalarType BFloat16" (numpy has no bf16) — mechanism right, patch incomplete |
| `logs/wrap_7186727.log`, `logs/nvidia-smi_7186727.txt`, `logs/7186727.*.OU/.ER` | 7186727 | debug | 1 GPU + V1 + `.cpu().float()` patch | Unsuccessful | Lost the startup EAGAIN race (hung; qdel + resubmit as 7186732) |
| `logs/wrap_7186732.log`, `logs/nvidia-smi_7186732.txt`, `logs/7186732.*.OU/.ER` | 7186732 | debug | same as 7186727, resubmitted | Unsuccessful | Patch cleared cuda+bf16 layers; third layer: receiver got a raw list (`'list' object has no attribute 'shape'`) — vLLM 0.8.4 V1 decoder only reconstructs type-hinted tensors. Conclusion: per-parameter V1 path is unserializable → route via NCCL broadcast (distinct GPUs) |
| `logs/wrap_7186739.log`, `logs/nvidia-smi_7186739.txt`, `logs/7186739.*.OU/.ER` | 7186739 | debug | roles on distinct GPUs, first try | Unsuccessful | Config parse error: `device_mapping` was a YAML list; ROLL `eval()`s it — must be a string (`"[0]"`) |
| `logs/wrap_7186742.log`, `logs/nvidia-smi_7186742.txt`, `logs/7186742.*.OU/.ER` | 7186742 | debug | distinct GPUs, strings fixed | Unsuccessful overall, but the BREAKTHROUGH run | `weight update progress: 100%` over NCCL broadcast (V1 serialization problem gone). Died in the step-0 *validation* cascade: the val RequestScheduler had been killed by the startup EAGAIN race; hung 45 min until qdel. Fix: skip val scheduler when `eval_steps > max_steps` |
| `logs/wrap_7186746.log`, `logs/nvidia-smi_7186746.txt`, `logs/7186746.*.OU/.ER`, `results/tictactoe_selfplay_polaris_smoke/7186746_*/logs/custom_logs.log` | 7186746 | debug | 0.5B deepspeed smoke, roles on GPUs 0/1/2, val disabled | **Successful — the GREEN bring-up** | 3 steps, `pipeline complete!`, 12 G `checkpoint-2`, TensorBoard events, `Training exited with code: 0` |
| `logs/wrap_7197416.log`, `logs/nvidia-smi_7197416.txt`, `logs/7197416.*.OU/.ER` | 7197416 | debug | Megatron-toolchain on-GPU probe (`scripts/polaris_megatron_probe.pbs`), new 10.25 G tarball | Successful | Rung 2: stack staged+imported on-node in 14.5 s, `flash_attn_func` bf16 forward + TE LayerNorm ran on the A100, stub dormant, pids.peak 83, Exit_status 0 |
| `logs/wrap_7197419.log`, `logs/pids_census_7197419.csv`, `logs/nvidia-smi_7197419.txt`, `logs/7197419.*.OU/.ER`, `results/tictactoe_selfplay_polaris_megatron/7197419_*/logs/custom_logs.log` | 7197419 | debug | Megatron 3-step smoke, layout B, attempt 1 | Unsuccessful | Lost the startup pids race: census `pids.peak` hit the 4096 ceiling during GPU-worker init; `actor_train-2` died at `ActorWorker.initialize()` with `RuntimeError: Resource temporarily unavailable` (NCCL watchdog EAGAIN cascade). Exited cleanly, code 1, ~4 min — no hang. First measured layout-B data point |
| `logs/wrap_7197421.log`, `logs/pids_census_7197421.csv`, `logs/nvidia-smi_7197421.txt`, `logs/7197421.*.OU/.ER`, `results/tictactoe_selfplay_polaris_megatron/7197421_*/logs/custom_logs.log` | 7197421 | debug | Megatron 3-step smoke, layout B, attempt 2 (unchanged resubmit) | Unsuccessful, but won the race and isolated the next blocker | Startup race won; HF→mca conversion measured FAST (~17 s/rank); vLLM EngineCore then refused to start: `max seq len (40960)` needs 5.62 GiB KV > 2.89 GiB available at util 0.35 — vLLM defaults `max_model_len` to Qwen3-4B's 40960 max_position_embeddings. Fix: `max_model_len: 8192` in the vLLM strategy_config. (Census pids columns read 0 on this node — wrap now derives the cgroup from `/proc/self/cgroup`) |
| `logs/wrap_7197423.log`, `logs/pids_census_7197423.csv`, `logs/nvidia-smi_7197423.txt`, `logs/7197423.*.OU/.ER`, `results/tictactoe_selfplay_polaris_megatron/7197423_*/logs/custom_logs.log` | 7197423 | debug | Megatron 3-step smoke, layout B, attempt 3 (max_model_len fix) | Unsuccessful | Lost the startup pids race again: census peak 3938/4096, `actor_train-0` EAGAIN at NCCL watchdog during `initialize()`. Race record now 1 win / 2 losses at layout B. Exited cleanly (code 1). Prompted the thread-trim package (RAY_NUM_CPUS 16→8, TORCH_NCCL_ENABLE_MONITORING=0, NCCL socket thread caps) + thread-owner census |
| `logs/wrap_7197427.log`, `logs/pids_census_7197427.csv`, `logs/thread_census_7197427.log`, `logs/nvidia-smi_7197427.txt`, `logs/7197427.*.OU/.ER`, `results/tictactoe_selfplay_polaris_megatron/7197427_*/` | 7197427 | debug | Megatron 3-step smoke, layout B, attempt 4 (trims + max_model_len) | **Successful — megatron GREEN** | 3 steps + `pipeline complete!`, megatron checkpoint `mp_rank_0{0..3}/model_optim_rng.pt` + `dist_optimizer` (14 G/rank), grad_norm 1.31→0.71, tps ~185–202, Exit_status 0, 11m54s. Census: pids.peak **4094/4096**, per-GPU maxima 23.3/25.2/22.1/21.8 GB. Thread census attributed 2120 threads to `ray::RequestScheduler` (its `multi_thread: 2048` Ray concurrency pool fills eagerly) |
| `logs/wrap_7197442.log`, `logs/pids_census_7197442.csv`, `logs/thread_census_7197442.log`, `logs/nvidia-smi_7197442.txt`, `logs/7197442.*.OU/.ER`, `results/tictactoe_selfplay_polaris_megatron/7197442_*/` | 7197442 | debug | **20-step proof run** (megatron layout B + RequestScheduler pool patch), `..._megatron_20step.yaml` | **Successful — the scale-up GREEN** | 20/20 steps, `pipeline complete!`, Exit_status 0, **37m37s**; incremental `checkpoint-9` (written mid-run) + final `checkpoint-19` (`mp_rank_0{0..3}` + `dist_optimizer`, 106 G total); census pids.peak **2313**/4096 (pool patch: was 4094), per-GPU maxima 23.6/25.3/22.3/22.0 GB; tps to 282 |
| `logs/wrap_7197445.log`, `logs/nvidia-smi_7197445.txt`, `logs/7197445.*.OU/.ER`, `results/tictactoe_selfplay_polaris_smoke/7197445_*/` | 7197445 | debug | 0.5B deepspeed smoke regression from the NEW megatron tarball + RequestScheduler pool patch | Successful | GREEN baseline intact after the env moved: 3/3 steps, `weight update progress: 100%` each step, `pipeline complete!`, `checkpoint-2` written (pruned to listing proof), Exit_status 0, 6m22s — and it won the startup race first try with the pool patch in effect |

## Decisions / changes log — megatron scale-up

- **2026-06-12 — Orientation + plan.** Read the proven artifacts (notes, guide,
  Midway megatron trio, Polaris smoke trio, mcore_adapter, Dockerfile);
  re-verified queue limits via `qstat -Qf` (debug-scaling caps at 1 h — same as
  debug; preemptable is the >1 h option); confirmed Qwen3-4B geometry from
  config.json (32/8 heads); code-verified the weight-sync path (NCCL bucket
  broadcast, no P2P with 4 TP src ranks — see Placement engineering). Decided
  layout B as primary. Drafted the ALCF pids.max ticket (above).
- **2026-06-12 — Venv extension started.** flash-attn 2.7.2.post1 cp312 wheel
  installed; megatron-core 0.12.3 installed after the pip-resolver incident
  (torch 2.12/cu13 clobber — see Toolchain extension; recovered, regression
  green); transformer-engine 2.2.0 building (first build OOM-killed by the
  login node's 8 GiB user cgroup at MAX_JOBS=16; retrying at MAX_JOBS=2).
  Authored the Polaris megatron trio (`agentic_val_tictactoe_selfplay_polaris_megatron.yaml`,
  `run_agentic_pipeline_tictactoe_selfplay_polaris_megatron.sh`,
  `scripts/train_polaris_megatron.pbs` — with a pids+GPU-memory census loop
  baked into the wrap). mcore_adapter will resolve from the venv
  (pip-installed, in the tarball), not from `mcore_adapter/src` on PYTHONPATH;
  the repo's `mcore_adapter/` dir cannot shadow it (no `__init__.py` → only a
  namespace-package candidate, regular packages win).
- **2026-06-12 — transformer-engine 2.2.0 installed; runtime library-resolution
  fix found.** The MAX_JOBS=2 rebuild succeeded (~25 min;
  `transformer_engine/`+`_cu12`/`_torch` all 2.2.0; pins re-verified intact).
  Two import-time findings, both measured on the login node and folded into the
  wrap/probe scripts:
  * `_load_nvrtc()` needs `CUDA_HOME` set (it globs there first; its ldconfig
    fallback fails where `ldconfig` is not on PATH). All wraps already set
    `CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1`.
  * `libtransformer_engine.so` links `libcudnn_adv.so.9`/`libcublas.so.12`
    directly; torch preloads the *main* libcudnn/libcublas but not the cuDNN
    sub-libraries, so `import transformer_engine` fails unless the venv's own
    `site-packages/nvidia/{cudnn,cublas,cuda_nvrtc,cuda_runtime,nccl}/lib`
    dirs are on `LD_LIBRARY_PATH`. Added to `train_polaris_megatron.pbs` and
    `polaris_megatron_probe.pbs` (these are torch's bundled cu124 libs, so the
    "torch's libs must win at runtime" rule is preserved; the /soft toolkit's
    lib64 stays OFF the path). With that, `torch + transformer_engine(.pytorch)
    + flash_attn + megatron.core` all import cleanly on the login node.
  Also authored `scripts/polaris_megatron_probe.pbs` (validation rung 2: stage
  new tarball, on-node imports, minimal flash-attn forward, stub-dormancy
  assert, cgroup pids dump) and validated the new megatron YAML offline (hydra
  `compose()` + `from_dict(AgenticConfig, ...)`: device_mapping strings eval to
  `[0,1,2,3]`/`[0]`/`[1]`, megatron strategy_config parses). apex 25.04
  building (`--cpp_ext --cuda_ext`, MAX_JOBS=2, sm_80 only).
- **2026-06-12 — apex + mcore_adapter installed; rung 1 (login validation)
  passed.** The apex build completed after ~25 min (`Successfully installed
  apex-0.1`; all CUDA extensions import — see Toolchain extension).
  `mcore_adapter 0.6.0.dev0` installed `--no-deps` from the repo; hit and
  fixed the **setuptools-82/pkg_resources regression** (downgraded to
  setuptools 75.8.2 rather than patch vendored source) and backfilled
  megatron-core's real runtime deps **zarr 2.18.7 + tensorstore 0.1.84**
  surfaced by `pip check` (dist_checkpointing backend; air-gap insurance) —
  details in Toolchain extension. Full rung-1 validation then passed in one
  process on the login node: megatron surface (megatron.core 0.12.3,
  transformer_engine 2.2.0 + .pytorch, flash_attn 2.7.2.post1, apex + amp_C +
  fused_layer_norm_cuda, mcore_adapter 0.6.0.dev0 incl.
  `mcore_adapter.trainer` and the real `RecvBucketManager`), stub dormancy
  (`vllm_strategy.RecvBucketManager.__module__ ==
  mcore_adapter.models.converter.convert_utils`),
  `MegatronTrainStrategy` import, proven-set regression (vllm 0.8.4, ray
  2.46.0, deepspeed 0.16.4, transformers 4.51.2, tokenizers 0.21.4,
  numpy 1.26.4, pyspiel, AgenticPipeline), and the hydra `compose()` +
  `from_dict` dry-run of the new megatron config (layout B confirmed:
  megatron_train TP=4 on `[0,1,2,3]`, vllm util 0.35 on `[0]`, hf_infer on
  `[1]`, Qwen3-4B, max_steps 3, env_groups 2). One trap for future login-node
  validation runs: the **thread caps are required on the login node too** —
  without `OMP/OPENBLAS/MKL_NUM_THREADS=1`, OpenBLAS tried to spawn 128
  threads inside the import chain, hit the login cgroup's `pids.max=256`
  (`pthread_create failed ... Resource temporarily unavailable`) and the
  process segfaulted in cv2's bootstrap. With caps set, the full surface
  imports in ~27 s.
- **2026-06-12 — apex + mcore_adapter installed; RUNG 1 (login-node toolchain)
  COMPLETE.** apex 25.04 (`e13873d`) built with `--cpp_ext --cuda_ext` at
  MAX_JOBS=2 / sm_80-only in ~50 min; `amp_C` and `fused_layer_norm_cuda` CUDA
  extensions import. `pip install --no-deps --no-build-isolation
  ./mcore_adapter` → 0.6.0.dev0. Full pin guard passed (all 12: torch 2.6.0,
  vllm 0.8.4, ray 2.46.0, deepspeed 0.16.4, transformers 4.51.2, tokenizers
  0.21.4, numpy 1.26.4, flash-attn 2.7.2.post1, megatron-core 0.12.3,
  transformer-engine 2.2.0, apex 0.1, mcore_adapter 0.6.0.dev0). ROLL surfaces:
  `roll...vllm_strategy` imports with the **real** `RecvBucketManager`
  (`mcore_adapter.models.converter.convert_utils` — stub dormant, asserted),
  `roll...megatron_strategy` full import surface OK, agentic pipeline OK.
  Two further findings while validating:
  * **The login node has its own pids wall.** Import tests without thread caps
    intermittently died (one segfault, OpenBLAS `pthread_create failed ...
    thread N of 64`): the login per-user cgroup is `pids.max=256`, and
    numpy/OpenBLAS's default 64-thread pool collides with it whenever a couple
    of processes import concurrently. With `OMP/OPENBLAS/MKL/NUMEXPR=1` the
    same imports pass deterministically (26 s warm for the whole ROLL+megatron
    chain). Mirror of the compute-node lesson; export the caps for ANY
    login-node python that imports numpy.
  * megatron.core guards its `import transformer_engine` with
    `except ImportError` only — TE's ldconfig `CalledProcessError` (when
    CUDA_HOME is unset) escapes the guard and kills the whole import. Another
    reason CUDA_HOME must always be set (all wraps do).
  Tarball `marshal-train-megatron-venv.tar` packing from the 9.6 GB venv
  (was 8.7 GB); `marshal-train-venv.tar` (8.3 GB, GREEN fallback) untouched.
- **2026-06-12 — RUNG 2 (on-GPU toolchain probe) PASSED — job 7197416.**
  First job on the new `marshal-train-megatron-venv.tar`: extraction 16 s,
  torch CUDA probe rc=0, the full megatron surface imported on-node in 14.5 s,
  `flash_attn_func` (bf16, causal) and a TE LayerNorm executed on the A100,
  and the on-node assert confirmed `vllm_strategy.RecvBucketManager` resolves
  from `mcore_adapter.models.converter.convert_utils` (stub dormant).
  pids: probe-only peak 83 of 4096; the job cgroup exposes `pids.peak` here,
  so the smoke's census will record true peaks. Exit_status 0, ~3 min of
  walltime. Proceeding to rung 3 (the layout-B megatron smoke).
- **2026-06-12 — Megatron smoke attempt 1 (job 7197419): lost the startup pids
  race; first measured layout-B census.** All pre-flight phases passed (16 s
  staging, probes green, Ray up with `CPU: 16.0`). During GPU-worker
  construction the census climbed 1512 → 3936 and `pids.peak` hit **4096**
  (the ceiling) — `actor_train-2` died at `initialize()` with EAGAIN and the
  NCCL watchdogs cascaded. The job *exited* (code 1, ~4 min) instead of
  hanging — cheaper to retry than the deepspeed smoke's hangs. Measurement:
  layout B's startup herd uses the whole 4096 budget; steady-state before GPU
  init was ~1030. Resubmitted unchanged as job 7197421 (single-variable: race
  variance), with thread-trim options held in reserve (RAY_NUM_CPUS 16→8,
  NCCL thread knobs) if the race keeps losing.
- **2026-06-12 — Megatron smoke attempt 2 (job 7197421): race won; next
  deterministic blocker found and fixed (vLLM max_model_len).** The resubmit
  cleared worker construction and megatron init — and answered the
  conversion-time question: mcore_adapter's HF→mca conversion logged `End
  loading, cost: ~17 s` per rank (page-cached eagle reads + in-memory
  conversion; NOT a walltime concern; it re-runs per job since the model dir
  has no mca-format checkpoint). vLLM then died at engine init:
  `ValueError: To serve at least one request with the model's max seq len
  (40960), 5.62 GiB KV cache is needed, which is larger than the available KV
  cache memory (2.89 GiB)` — Qwen3-4B ships `max_position_embeddings: 40960`,
  vLLM 0.8.4 defaults `max_model_len` to it, and at util 0.35 (14 GB) only
  2.89 GiB remains for KV after the 8 GB weights. Pipeline sequences are
  <=~5 k tokens, so the fix is `max_model_len: 8192` in the strategy_config
  (passes through `**vllm_config` to the engine; 2.89 GiB ≈ 21 k cached
  tokens at 144 KB/token). Chosen over raising gpu_memory_utilization, which
  would shrink GPU0's margin against the colocated TP rank-0 shard. Also:
  the census read pids 0 on this node — the wrap now derives the cgroup path
  from `/proc/self/cgroup` instead of assuming `/sys/fs/cgroup/jobs/<jid>`.
  GPU memory columns DID record: GPU0 9.77 GB (vLLM weights+overhead),
  GPUs 1-3 ~1.5 GB at death. Resubmitting with the max_model_len fix.
- **2026-06-12 — Megatron smoke attempt 3 (job 7197423): race lost again;
  stopped re-rolling and trimmed the startup herd.** Same signature as
  attempt 1 (census peak 3938 of 4096; `actor_train-0` EAGAIN in NCCL watchdog
  at `initialize()`). Race record at layout B: 1 win in 3. Code audit before
  the next attempt: (a) role inits are already serialized
  (`agentic_pipeline.py`: actor_train → ray.get → actor_infer blocking →
  reference blocking), so the peak is actor_train's 4 TP ranks initializing
  concurrently — which cannot be serialized (they form collectives);
  (b) ROLL worker actors reserve `num_cpus=0.01` (cluster.py:132), so
  shrinking `RAY_NUM_CPUS` cannot starve actor scheduling; (c) ROLL's
  `RequestScheduler` declares a Ray concurrency group allowing up to **2048**
  threads (generate_scheduler.py:756; GenerateScheduler 128/256) — these
  pools fill under request load, a rollout-phase hazard to watch, not the
  startup killer. Trims applied to the megatron launcher: `RAY_NUM_CPUS=8`
  (halves the idle-worker pool, ~250 threads), `TORCH_NCCL_ENABLE_MONITORING=0`
  (drops one monitor thread per process group across the many megatron PGs),
  `NCCL_SOCKET_NTHREADS=1`/`NCCL_NSOCKS_PERTHREAD=1`. The wrap gains a
  thread-owner sampler (`logs/thread_census_<jid>.log`, `ps -o nlwp` top-12
  every 30 s) so any further loss is attributable per-process.

## GREEN — megatron_train path passed on Polaris — Successful, 2026-06-12

**The configuration MARSHAL's real experiments use closed its training loop
end-to-end on ALCF Polaris**: Qwen3-4B, `actor_train: megatron_train` with
TP=4 + sequence_parallel + distributed optimizer + recompute=full, vLLM
rollouts, frozen hf reference — the Polaris analog of Midway's GREEN jid
50261211, on 40 GB A100s instead of 140 GB H200s.

| Field | Value |
|---|---|
| PBS jid | **7197427** (`debug`), node `x3204c0s37b0n0` |
| Config | `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_megatron.yaml` — layout B: actor_train TP=4 `"[0,1,2,3]"`, actor_infer vLLM `"[0]"` (util 0.35, `max_model_len: 8192`), reference `"[1]"`, env_groups 2, val disabled |
| Steps | 3/3 — `pipeline step 0/1/2 finished` → `pipeline complete!` |
| Walltime | **11m54s total** (14 s tarball staging, ~15 s/rank HF→mca conversion, 3 steps ≈ 95–110 s each, 53 G checkpoint write); Exit_status 0; wrap `Training exited with code: 0` |
| Metrics | `actor_train/grad_norm` 1.31 → 0.71, `actor/kl_loss` 0.0031, `system/tps` 185–202; vLLM KV cache 21,008 tokens |
| Checkpoint | `results/tictactoe_selfplay_polaris_megatron/7197427_20260612-075718/actor_train-{0..3}/checkpoint-2/iter_0000001/mp_rank_0{0..3}/model_optim_rng.pt` + `dist_optimizer/` (14 G/rank, 53 G run dir) + `pipeline/checkpoint-2/` — the mcore_adapter save path, distinct from deepspeed's `bf16_zero_pp_rank_*` |
| **pids census (measured)** | steady-state ~1030 pre-GPU-init → **peak 4094 of 4096** during construction; survived by 2 threads. `logs/pids_census_7197427.csv` (140 rows) |
| **Memory census (measured)** | per-GPU maxima 23.3 / 25.2 / 22.1 / 21.8 GB of 40 GB — layout B's 18 GB/rank + colocated role math holds with >14 GB margin |
| Weight-sync | megatron→vLLM bucket path over NCCL broadcast: `model_update_end_onload/offload` every step; rollouts generated correctly after each sync (no `weight update progress` tqdm on this path — that line is the deepspeed per-parameter sync's) |
| Thread attribution | `logs/thread_census_7197427.log`: **`ray::RequestScheduler` owns 2120 threads** — Ray eagerly fills its `multi_thread: 2048` concurrency-group pool (generate_scheduler.py:756). Single largest pids consumer; also explains the 0.5B smoke's ~50% race (same pool + fewer workers) |

Four attempts to GREEN: 7197419 (race loss), 7197421 (race won; found vLLM
`max_model_len` 40960 default blocker; census path bug), 7197423 (race loss →
trim package), 7197427 (GREEN at 2 threads of margin).

### Reproduce
```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/MARSHAL
qsub scripts/train_polaris_megatron.pbs   # defaults to marshal-train-megatron-venv.tar
```
- **2026-06-12 — RequestScheduler pool patch verified live (job 7197442,
  20-step run).** With `generate_scheduler.py`'s `multi_thread` concurrency
  group reduced 2048 → 256, the 20-step run's startup peak measured
  **2177 of 4096** (vs 4094 on the GREEN smoke) and the RequestScheduler
  process owns 328 threads (vs 2120). The startup EAGAIN race that consumed
  attempts 7197419/7197423 (and ~half of all 0.5B-smoke submissions during
  the bring-up) is resolved by an attributable, in-repo, one-line change —
  not by raising the cap. The ALCF ticket remains worth filing as hygiene
  for larger future layouts.

## GREEN — 20-step megatron proof run — Successful, 2026-06-12

The Polaris analog of Midway's GREEN scale-up (jid 50259767), completing the
mission's Phase 2: same layout-B megatron config as the 3-step GREEN, only
`max_steps` 3 → 20 and `save_steps` 3 → 10 changed (single-axis discipline),
plus the RequestScheduler pool patch.

| Field | Value |
|---|---|
| PBS jid | **7197442** (`debug`) |
| Config | `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_megatron_20step.yaml` |
| Steps | **20/20** — `pipeline step 0..19 finished` → `pipeline complete!` |
| Walltime | **37m37s** (init+first step 7m37s; steady state ~75–80 s/step; predicted ~47 min worst case vs debug's 60 min cap). Exit_status 0; wrap `Training exited with code: 0` |
| Checkpoints | **Incremental saving proven**: `checkpoint-9` written mid-run (on disk at 08:34, verified while the run continued) and `checkpoint-19` at completion — both megatron-format `actor_train-{0..3}/.../iter_0000001/mp_rank_0{0..3}/model_optim_rng.pt` + `dist_optimizer/` (106 G total) + `pipeline/checkpoint-{9,19}` |
| Metrics | `system/tps` to 282; non-trivial `actor_train/grad_norm` through step 19; TensorBoard events in `results/.../7197442_20260612-081255/tensorboard/` |
| **pids census** | peak **2313 of 4096** across the whole run (445 census rows) — the RequestScheduler patch (2048→256) cut the startup peak from the smoke's 4094; `RequestScheduler` owns 328 threads (was 2120) |
| Memory census | per-GPU maxima 23.6 / 25.3 / 22.3 / 22.0 GB of 40 GB over 20 steps — no growth trend vs the 3-step run |
| Queue decision | `debug` (1 h) chosen over `preemptable` because the measured per-step time fit with margin; preemptable + `-r y` + low `save_steps` remains the documented path for runs that exceed ~50 min |
- **2026-06-12 — 20-step proof run GREEN (job 7197442) + checkpoint hygiene.**
  Phase 2 complete — see the "GREEN — 20-step megatron proof run" section.
  Queue decision shown by the math: measured init 7m37s + ~78 s/step + ~2 min/
  save → 20 steps fit `debug` (37m37s actual vs 60 min cap); `preemptable`
  (≤72 h, `-r y`, low save_steps, resume_from_checkpoint) is the documented
  path past ~35 steps. After verification, heavy checkpoint blobs were pruned
  per the established practice: 7197427's `checkpoint-2` (53 G) and 7197442's
  `checkpoint-9`+`checkpoint-19` (106 G) replaced by
  `checkpoint_listing_proof.txt` (full `ls -laR` + `du`) in each run dir;
  TensorBoard, logs, and `pipeline/checkpoint-*` retained. The deepspeed GREEN
  smoke's 12 G `checkpoint-2` (job 7186746, verified 2026-06-06) was pruned the
  same way. A 0.5B deepspeed regression from the NEW tarball + the pool patch
  is running as job 7197445 (the shared env moved; the GREEN baseline must not
  silently rot) — result recorded in the ledger.
- **2026-06-12 — 0.5B deepspeed baseline regression PASSED from the new tarball
  (job 7197445).** The proven smoke config, staged from
  `marshal-train-megatron-venv.tar`, with the real `RecvBucketManager` import
  active and the RequestScheduler pool patch: 3/3 steps, per-parameter NCCL
  weight-sync at 100% each step, `pipeline complete!`, Exit_status 0 in
  **6m22s** (vs ~8 min historically), winning the startup race on the first
  submission. Both training strategies (deepspeed_train and megatron_train)
  are now proven on Polaris from the same extended venv, and the GREEN
  baseline did not rot. Checkpoint pruned to `checkpoint_listing_proof.txt`
  per the established practice.
