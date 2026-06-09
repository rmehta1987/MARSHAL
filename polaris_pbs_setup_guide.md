# MARSHAL on Polaris (PBS) — step-by-step setup guide

A clean, do-this-in-order recipe for standing up the MARSHAL tic-tac-toe self-play
pipeline on ALCF **Polaris** (PBS Pro). This is the distilled "how to reproduce it"
companion to `polaris_pbs_notes.md` — the notes file is the chronological decisions
log (every failure, in the order we hit it); **this file is the procedure** with the
dead-ends folded into sub-bullets so you don't relearn them.

> Status when written: the smoke is **GREEN** (jid 7186746) — 3 DeepSpeed REINFORCE
> steps → `pipeline complete!` → 12 G checkpoint + TensorBoard, exit 0. The steps
> below are exactly what produces that.

**Conventions used below**

- `$BASE` = `/lus/eagle/projects/lighthouse-uchicago/members/mehta5`
- `$REPO` = `$BASE/MARSHAL` (the project root, where you run `qsub`)
- Sub-bullets marked **✗ Didn't work:** record a thing we tried that failed and why —
  skip past them, they're there so you don't repeat them.

---

## Step 0 — Know the cluster facts before you touch anything

| Item | Value |
|---|---|
| Scheduler / login | PBS Pro (`qsub`/`qstat`/`qdel`), `polaris-login-02` |
| Account (`-A`) | **`lighthouse-uchicago`** |
| Queue | `debug` (1–2 nodes, ≤1 h) for the smoke |
| GPU | 4× A100 **40 GiB** per node (sm_80) |
| CPU/RAM | EPYC Milan 32c/64t (`ncpus=64`), 512 GiB |
| Native CUDA | 12.4.1 at `/soft/compilers/cudatoolkit/cuda-12.4.1` |
| Conda module | `conda/2025-09-25` (base py 3.12.11) |
| Filesystems | must pass `-l filesystems=home:eagle` or PBS rejects the job |

- **✗ Didn't work: `-A Uchicago-lighthouse`.** PBS rejects it (`Project ... not
  found`). The real allocation name is `lighthouse-uchicago` — confirm with
  `sbank-list-allocations` if in doubt.
- **✗ Didn't work: omitting `-l filesystems=home:eagle`.** Polaris refuses the job
  outright. Every `qsub` must declare the filesystems it touches.
- **✗ Didn't work: the Apptainer/container route from the handoff.** No `.sif` exists
  on Polaris, the Aliyun ROLL registry is unreachable from ALCF, and `container_extras/`
  is `.gitignore`'d. We go **native venv** instead (Step 1).

---

## Step 1 — Build the training venv (one time, on the login node)

The login node has outbound network + HF reachability; pip needs no GPU. Install
**ROLL's exact pinned stack** — the same versions the official container froze — so
the ray/vllm version conflict that killed the Midway source-install can't happen
(vllm 0.8.4 is happy with ray 2.46.0).

```bash
module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base
export PIP_CACHE_DIR=$BASE/pip_cache TMPDIR=$BASE/tmp
python -m venv $BASE/conda-envs/marshal-train        # clean — NO --system-site-packages
source $BASE/conda-envs/marshal-train/bin/activate
pip install --upgrade pip wheel setuptools

# 1) torch first (pins the cu124 build): torch 2.6.0 -> cuda 12.4, triton 3.2.0
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# 2) the consistent inference/RL trio (joint resolve)
pip install vllm==0.8.4 ray==2.46.0 deepspeed==0.16.4

# 3) pin ROLL's tested versions + smoke extras (this DOWNGRADES the bleeding-edge
#    transformers/numpy that step 2 dragged in)
pip install transformers==4.51.2 "tokenizers<0.22" "numpy<2.0" \
            datasets==3.1.0 peft==0.12.0 accelerate==0.34.2 \
            tensordict modelscope "tyro>=0.5.7" pydantic loralib einops isort jsonlines \
            deprecated dacite codetiming more_itertools wandb math-verify hydra-core omegaconf \
            gym "gymnasium[toy-text]" gym_sokoban "trl>=0.11,<0.19" \
            open_spiel matplotlib tensorboard
```

Target resolved set: `torch 2.6.0+cu124 · vllm 0.8.4 · ray 2.46.0 · deepspeed 0.16.4
· transformers 4.51.2 · tokenizers 0.21.4 · numpy 1.26.4 · open_spiel 1.6.15`.
Verify with `python -c "import torch, vllm, ray, deepspeed, pyspiel; print('ok')"`.

- **✗ Didn't work: `pip install -r requirements*.txt` blindly.** ROLL's requirement
  files and the vllm/ray/deepspeed trio pull **transformers 5.x / numpy 2.x** — the
  exact "transformers must be `<5`" hazard (5.x removed `all_special_tokens_extended`).
  Step 3 must run *after* step 2 to claw the pins back down.
- **✗ Didn't work: using ALCF's base-conda torch (2.8.0).** ALCF docs recommend not
  pip-installing a custom torch, but ROLL/vllm 0.8.4 *pin* `torch==2.6.0`. We
  deliberately deviate. This is safe because cu124 is Polaris's native CUDA and the
  smoke is single-node (no ALCF multi-node fabric needed).
- **Harmless:** pip warns `cupy-cuda12x` / `opencv-python-headless` want numpy≥2.
  Neither is imported on the text-game + deepspeed path. Ignore (or pin
  `cupy-cuda12x<14` / `opencv-python-headless<4.10` only if a future path imports them).
- **megatron/transformer-engine/apex/flash-attn are intentionally NOT installed.** The
  smoke uses `deepspeed_train` + `attn_implementation: eager`, so it needs none of them.
  (One consequence is handled by a source stub — Step 3.)

---

## Step 2 — Pack the venv into ONE tarball (the load-bearing Polaris fix)

**This is the single most important Polaris-specific step.** A venv left on eagle
(Lustre) is catastrophically slow to import on a compute node, because `import torch`
opens thousands of small `.so`/`.py` files and Lustre is terrible at that metadata
storm. Pack it into one big file and stage it to node-local SSD per job instead.

```bash
cd $BASE/conda-envs
tar cf $BASE/marshal-train-venv.tar -C marshal-train .   # ~8.3 G, one-time ~20 min
```

The PBS wrap (Step 4) reads this one file (~20 s) and extracts to `/local/scratch`
(~12 s), then runs python from there.

- **✗ Didn't work: importing the venv directly off eagle.** `import torch` on a cold
  compute node **hung past 300 s** (vs 47 s on the login node). Measured root cause:
  one 988 MB sequential read off eagle = 3 s (fine), but copying the venv's **71,700
  files** off eagle = **1137 s**. Big files fast, many small files fatal.
- **✗ Didn't work: putting the venv on `/home`.** It fits (45 G quota) but `/home` is
  *also* Lustre — same small-file problem. `/soft` is fast but read-only. **Only
  node-local SSD is both fast and writable**, hence per-job staging from a tarball.
- **✗ Didn't work: `source $VENV/bin/activate` on the staged copy.** A relocated
  venv's `activate` hardcodes the *original* build path. The wrap activates manually
  (prepend `$VENV/bin` to `PATH`, set `VIRTUAL_ENV`) so the location-independent
  `pyvenv.cfg` drives `sys.path`. Never source a relocated venv's activate.

---

## Step 3 — Apply the source patches (already in this repo; verify they're present)

These live in `roll/` (imported from the repo via `PYTHONPATH`, **not** from the venv
tarball — so editing source needs **no tarball rebuild**, just the next `qsub`).

1. **`RecvBucketManager` stub — `roll/distributed/strategy/vllm_strategy.py`.** That
   file unconditionally imports `RecvBucketManager`, which pulls in `megatron.core` at
   load. With megatron-core absent (deepspeed smoke), importing `VllmStrategy` would
   `ModuleNotFoundError`. The fix wraps the import in `try/except ImportError` with a
   minimal stub whose `process_bucket()` raises `NotImplementedError` (that path is the
   megatron→vllm weight-sync, unreachable with a deepspeed actor).

2. **Node-IP resolution — `roll/distributed/executor/worker.py`.** ROLL's
   `get_node_ip()` resolved the worker IP by opening a UDP socket to `8.8.8.8:80`.
   **Polaris compute nodes are air-gapped** → `OSError: Network is unreachable` → every
   worker dies at construction. The fix resolves via `ray.util.get_node_ip_address()`
   first, falling back to the old trick only if that fails.

3. **V1 weight-sync defense — `roll/third_party/vllm/vllm_0_8_4/llm.py:update_parameter`.**
   `.cpu().float()` any tensor before `collective_rpc` under vLLM V1 (defense for the
   P2P path; the real fix for the smoke is distinct-GPU placement in Step 5).

- **Did NOT need: the `log_monitor gcs_publisher` shim** the Midway source-install
  required. That was a ray ≥2.48 API change; at the pinned ray 2.46.0, ROLL's
  `log_monitor.py` works unmodified.

---

## Step 4 — Stage the three Polaris job files (already in this repo)

| File | Role |
|---|---|
| `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_smoke.yaml` | hydra config (Qwen2.5-0.5B, ZeRO-2, 3 steps, tensorboard, roles on distinct GPUs) |
| `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_polaris.sh` | in-job launcher (Ray cleanup, thread caps, env knobs) |
| `scripts/train_polaris.pbs` | PBS wrap (`#PBS` directives, modules, venv staging, caches, probe) |

Things the wrap/launcher get right that you must not regress:

- **Thread caps, exported BEFORE python:** `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=
  MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=VECLIB_MAXIMUM_THREADS=RAYON_NUM_THREADS=1`,
  `TOKENIZERS_PARALLELISM=false`. Every Ray worker must inherit these before its first
  numpy/OpenBLAS import.
- **`RAY_NUM_CPUS=16`** — shrinks Ray's idle-worker pool ~4× (default prestarts ~1
  per core = 64, each ~30+ threads).
- **`MAX_JOBS=4`** — caps ninja's compile burst when deepspeed JIT-builds `fused_adam`.
- **`VLLM_USE_V1=1`** — keep the V1 engine.
- **`CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1`** and **`CC=gcc-12 / CXX=g++-12`**
  for the deepspeed `fused_adam` JIT build.
- **`TORCH_EXTENSIONS_DIR=$BASE/torch_extensions`** + **`TORCH_CUDA_ARCH_LIST=8.0`** —
  compile `fused_adam` once, cache on eagle, every later job loads the `.so`.

What each of those *fixes* (these are the dead-ends, as sub-bullets):

- **✗ Didn't work: leaving CPU thread pools at default.** Each of the colocated workers
  spawned 64 OpenBLAS + 64 OMP threads; the combined total blew past the job cgroup's
  `pids.max=4096` → `OpenBLAS blas_thread_init: pthread_create failed ... Resource
  temporarily unavailable` and workers died at construction. **Not OOM, not `ulimit -u`
  (that's ~2 M) — the cgroup thread cap, which we can't raise from inside the job.**
- **✗ Didn't work: capping threads at 4 instead of 1.** Even 4/process still hit EAGAIN
  during the `fused_adam` compile burst (`gcc-12: cannot execute 'cc1plus': vfork:
  Resource temporarily unavailable`). Dropped to 1.
- **✗ Didn't work: `CC=gcc-14` (the conda module default).** CUDA 12.4 `nvcc` caps the
  host compiler at ≤13.2 and rejects gcc-14. gcc-12 is ≥9 (deepspeed's floor) and ≤13.2.
- **✗ Didn't work: forcing `VLLM_USE_V1=0`** to dodge a V1 serialization bug. V0 spawns
  *more* startup threads than V1, so under `pids.max=4096` V0 died at
  `RolloutScheduler.__init__` *before* training, while V1 reaches the loop. Keep V1 and
  fix the serialization a different way (Step 5).
- **✗ Didn't work: `nvidia-smi` in the foreground.** It hung in an unkillable D-state on
  multiple debug nodes, even under `timeout`. The wrap runs it fully detached
  (diagnostic only) and uses a separate fail-fast torch CUDA probe
  (`scripts/polaris_gpu_probe.py`, bounded by `timeout 300`, exit 42 on failure) so a
  wedged GPU can't burn the whole walltime.

---

## Step 5 — The config knobs that make it fit (distinct-GPU placement)

In `agentic_val_tictactoe_selfplay_polaris_smoke.yaml`, two settings are doing the
heavy lifting and are easy to get wrong:

- **`num_gpus_per_node: 4`** with **roles on DISTINCT GPUs**:
  `actor_train device_mapping: "[0]"`, `actor_infer: "[1]"`, `reference: "[2]"`. One
  Ray worker per role = 3 GPU workers total, which fits under `pids.max=4096`.
- **`env_groups: 2`** (and `n_groups: [2]`) for both train and val env managers.
- **vLLM `gpu_memory_utilization: 0.3`** (A100 40 GiB vs H200 140 GiB; ample for 0.5B).

Why each, with the dead-ends:

- **✗ Didn't work: `num_gpus_per_node: 4` with 4 ranks/role (the original).** ~12 GPU
  workers × hundreds of threads each → past `pids.max=4096` → EAGAIN. Shrinking to 1
  worker/role was required.
- **✗ Didn't work: colocating actor_train + actor_infer on the SAME GPU** (the obvious
  1-GPU way to save processes). When src and tgt share `(node, gpu)`, ROLL's comm-plan
  takes the **P2P** weight-sync path (`update_parameter.remote(weight=<cuda tensor>)`),
  and vLLM 0.8.4's **V1** engine msgpack-serializes RPC args across processes. The CUDA
  tensor can't round-trip — the error walked cuda → bf16 → `'list' object has no
  attribute 'shape'` as we patched each layer. **The per-parameter V1 path is simply
  not serializable.** Putting train and infer on *different* GPUs makes the comm-plan
  use **NCCL broadcast** (`collective.broadcast`, no msgpack) — V1-safe, no tensor ever
  serialized. Same pids footprint (still 3 workers), just spread across 3 of the 4 A100s.
- **✗ Didn't work: `env_groups: 16` (the original).** That spawns one RequestScheduler +
  `_QueueActor` per env instance = ~80 envs = ~190 Ray processes on one node →
  pids.max exhaustion. 2 groups → ~10 envs → ~35 processes → headroom. (Smoke only
  needs the loop to close; this just runs more trajectories per env sequentially.)
- **✗ Didn't work: `device_mapping: [0]`** as a YAML list. ROLL `eval()`s
  `device_mapping`, so it must be a **string**: `device_mapping: "[0]"`.
- **Also patched out:** the step-0 validation. `0 % eval_steps == 0` fired an implicit
  eval whose val RolloutScheduler was an extra startup actor (more pids pressure) and a
  failure path. `agentic_pipeline.py` now creates the val scheduler only when
  `eval_steps <= max_steps`, so the smoke skips it.

---

## Step 6 — Submit and watch

```bash
cd $REPO
qsub -v MARSHAL_VENV_TARBALL=$BASE/marshal-train-venv.tar scripts/train_polaris.pbs

# Watch live (PBS only flushes .OU/.ER at job END — use the wrap's live log instead):
tail -f logs/wrap_<jid>.log
tail -f results/tictactoe_selfplay_polaris_smoke/<jid>_*/logs/custom_logs.log
```

**Success looks like:** `weight update progress: 100%` (NCCL broadcast) →
`pipeline step 0/1/2 finished` → **`pipeline complete!`** → a 12 G
`actor_train-0/checkpoint-2/` on disk + a TensorBoard `events.out.tfevents...` file →
wrap log prints **`Training exited with code: 0`**.

- **Known residual flakiness (retry, don't debug):** the startup "thundering herd"
  (~17 processes importing torch/vLLM at once) peaks near `pids.max=4096` and **loses
  ~half the time** at `RolloutScheduler.__init__` with EAGAIN. A lost run *hangs* —
  `qdel <jid>` and resubmit. The durable fix is an **ALCF ticket to raise the per-job
  cgroup `pids.max`** (see scale-up).
- **✗ Didn't work: relying on `qdel` to promptly reap a hung job.** A process wedged in
  D-state (hung GPU / nvidia-smi) keeps the job `R` and burns walltime even after
  `qdel` returns 0. Nothing to do but let it hit walltime or escalate the node.
- **Infra note:** during the bring-up, many debug nodes were offlined with Lustre
  mount / prologue-hook failures (0-byte output, full-hour walltime kills). That's an
  **ALCF outage, not your setup** — `pbsnodes -l` to check; wait for recovery rather
  than resubmitting into it.

---

## Scaling to a more complex LLM (Qwen3-4B + Megatron)

The smoke proved the loop on a 0.5B model with one GPU per role. The real run uses
**~8× the parameters** and a model that **won't fit on one A100**, which changes the
shape of the job. (Plain-language companion: `polaris_scaleup_gpu_needs.md`.)

- **Model split across all 4 GPUs of a node (tensor/pipeline parallel), not 1-per-role.**
  Qwen3-4B is too big for a single 40 GiB A100, so the `megatron_train` strategy shards
  one model across all 4 GPUs that act as one. Reference config to port:
  `examples/tictactoe/agentic_val_tictactoe_selfplay_midway_megatron.yaml` (it was sized
  for Midway's 140 GiB H200s — **expect to turn dials down** for 40 GiB A100s).
- **The hard blocker is NOT the GPUs — it's `pids.max=4096`.** Ganging 4 GPUs per role
  means ~4× the worker processes/threads of the smoke, which sails past the cap. Before
  the scale-up can run reliably, **one of these must happen:**
  - **File an ALCF ticket to raise the per-job cgroup `pids.max`** on the GPU nodes.
    This is the clean fix (it's their setting) and the single thing most likely to
    unblock the scale-up. Do this *first* — everything else waits on it.
  - **Or trim the process/thread count further** (fewer parallel envs, leaner settings)
    to squeeze under 4096 — possible but fiddly and limiting.
- **The megatron toolchain must be installed** (deferred for the smoke):
  `megatron-core`, `transformer-engine`, `apex`, and the local `mcore_adapter/` package
  (`pip install ./mcore_adapter`), plus likely `flash-attn`. Once present, **remove the
  `RecvBucketManager` stub reliance** (the real module loads), and add `mcore_adapter/src`
  back to `PYTHONPATH` in the wrap. Rebuild the venv tarball after these adds.
- **Memory will be tight on one node.** 4B + train copy + optimizer state + a vLLM
  inference copy + a frozen reference, across 4×40 GiB = 160 GiB total. The setup leans
  on roles **taking turns** (offload/onload) so they don't all need room at once. First
  attempts will likely need: smaller `per_device_train_batch_size`, ZeRO-3 +
  CPU-offload, lower vLLM `gpu_memory_utilization`, shorter `sequence_length`.
  - **Fallback if one node won't fit: two nodes (8 GPUs).** Doubles memory and removes
    the squeeze, at the cost of `-l select=2:...` and cross-node NCCL. **This is where
    ALCF's multi-node AWS-OFI/NCCL fabric finally matters** — the smoke dodged it by
    being single-node; the megatron multi-node path needs it configured (ALCF docs warn
    a misconfigured fabric can hang Megatron-DeepSpeed). Fallback, not the starting point.
- **Suggested order:** (1) get `pids.max` raised; (2) run 4B on **one node / 4 GPUs**
  with megatron, lowering memory dials on the first one or two tries; (3) go to **two
  nodes** only if one won't fit; (4) *then* worry about long runs and perf tuning.

---

## Running it for a longer session

The smoke caps everything tiny (3 steps, `debug` queue, 1 h). A real training session
needs queue, walltime, checkpoint, and resume changes:

- **Move off `debug` to a production queue.** `debug` is 1–2 nodes / ≤1 h. Use:
  - `debug-scaling` — 1–10 nodes, but **1 job per user** and still short walltime;
    fine for a medium multi-node test.
  - `prod` (routing queue) — ≥10 nodes, up to **24 h** walltime, for the real run.
  - Update the `#PBS -q` and `#PBS -l walltime=` directives in `scripts/train_polaris.pbs`
    (or override at submit: `qsub -q prod -l walltime=24:00:00 ...`).
- **Raise the training length in the YAML:** `max_steps` (3 → hundreds/thousands),
  `save_steps` (checkpoint cadence — don't leave it at 3), `eval_steps`, `logging_steps`.
  Re-enable validation by setting `eval_steps <= max_steps` (the smoke patch disables
  the val scheduler when `eval_steps > max_steps`).
- **Checkpoint sizing & cleanup.** The 0.5B smoke wrote a **12 G** full DeepSpeed
  checkpoint (weights + ZeRO optimizer state) *per save*. A 4B model and frequent
  `save_steps` will fill eagle fast — set a sane `save_steps`, prune old checkpoints,
  and keep `ROLL_OUTPUT_DIR` on eagle (not node-local SSD, which is ephemeral).
- **Auto-resume across walltime kills.** The inherited Midway/CMU sbatch pattern wires
  `signal=B:SIGUSR1@90` + a `train_autoresume.sh` so the job checkpoints and requeues
  near the time limit. **PBS does the equivalent** with `qsub -W depend=afterany:<jid>`
  chaining or `#PBS -l walltime` + a trap on the PBS signal — port this before a run
  long enough to be killed. Set `resume_from_checkpoint: true` in the YAML so a requeued
  job continues instead of restarting.
- **Tracking.** The smoke uses `track_with: tensorboard` (offline-safe). For a long run
  you may want `wandb` (`track_with: wandb` + `WANDB_API_KEY`), but note compute nodes
  are **air-gapped** — wandb would need offline mode (`WANDB_MODE=offline`) + a later
  `wandb sync`, or stick with tensorboard.
- **Persisted caches help long/repeated runs.** `TORCH_EXTENSIONS_DIR` (cached
  `fused_adam.so`), `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, and `HF_HOME` are all
  on eagle and reused across jobs — keep them; they remove the compile/download bursts.
- **✗ Didn't work / watch out:** the startup EAGAIN pids race is *per job submission*,
  so a long run that gets requeued re-rolls that ~50% dice each time. Getting
  `pids.max` raised (scale-up section) matters even more for long unattended runs,
  otherwise a requeue can silently hang. Until then, monitor and resubmit on hang.

---

## Quick reference — the one command

```bash
cd /lus/eagle/projects/lighthouse-uchicago/members/mehta5/MARSHAL
qsub -v MARSHAL_VENV_TARBALL=/lus/eagle/projects/lighthouse-uchicago/members/mehta5/marshal-train-venv.tar \
     scripts/train_polaris.pbs
# tail -f logs/wrap_<jid>.log  ->  "pipeline complete!" + "Training exited with code: 0"
```

*Full failure-by-failure history and exact jids: `polaris_pbs_notes.md`. Plain-language
hardware/scale explainer: `polaris_scaleup_gpu_needs.md`.*
