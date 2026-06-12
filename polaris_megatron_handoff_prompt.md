# Handoff: scale MARSHAL's Polaris pipeline to the `megatron_train` path (Qwen3-4B on A100 nodes)

> Paste everything below the line into a fresh Claude Code session running on an
> ALCF **Polaris** login node, inside this repo checkout (branch
> `polaris-pbs-handoff`). It is written **to that Claude**. The deepspeed smoke
> (Qwen2.5-0.5B, 3 steps) is already proven GREEN on Polaris (jid 7186746); your
> job is to scale it to MARSHAL's real training path — **`megatron_train` with
> Qwen3-4B** — and to record the work to a standard a scientific reviewer would
> accept.

---

You are a Claude Code session on **ALCF Polaris** (PBS Pro: `qsub` / `qstat` /
`qdel`). The MARSHAL tictactoe self-play pipeline already closes its training
loop end-to-end here on one A100 node, via a native venv staged from a tarball.
Your mission is to take it from the bring-up toy (0.5B model, `deepspeed_train`,
one GPU per role) to the configuration MARSHAL's real experiments use.

## Mission

1. **Extend the training venv with the megatron toolchain** — `megatron-core`,
   the repo's local `mcore_adapter/`, `transformer-engine`, `apex`,
   `flash-attn` — *without* disturbing the proven torch 2.6.0+cu124 / vllm 0.8.4 /
   ray 2.46.0 stack, and rebuild the staging tarball. See Objective 1.
2. **Port the Midway-proven megatron config to Polaris.** The exact target shape
   was already validated on Midway (jid 50261211): Qwen3-4B, `actor_train:
   megatron_train` with TP + sequence_parallel + distributed optimizer +
   recompute=full, vLLM rollouts, 3 steps. Author Polaris copies; do not edit
   the Midway or GREEN-smoke files in place. See Objective 3.
3. **Solve the placement problem under the twin constraints.** A 40 GiB A100 is
   3.5× smaller than the H200s the megatron config was sized for, and the PBS
   job cgroup hard-caps the whole job at **`pids.max=4096`**
   threads+processes (measured, jid 7186710). Memory pushes you toward more
   GPUs per role; the pids cap pushes you toward fewer workers. Deriving and
   *verifying on the cluster* a layout that satisfies both is the load-bearing
   technical problem. See Objective 2.
4. **Escalate in two phases.** Prove the megatron path with a **3-step smoke on
   `debug`** (≤1 h, fast turnaround), then run a **20-step proof run** (the
   Polaris analog of Midway's GREEN scale-up, jid 50259767) on whatever queue
   its measured walltime requires. See Objective 4.
5. **Document to reviewer standard.** Keep `polaris_pbs_notes.md` (the lab
   notebook) and `polaris_pbs_setup_guide.md` (the recipe) current as you go —
   see **Documentation discipline**, which is a graded deliverable, not an
   afterthought.

## Do not go down these paths

- **The container/Apptainer route.** The original handoff assumed the official
  ROLL `.sif` is the unit of portability. On Polaris it is blocked — no image
  exists here, the Aliyun registry is unreachable from ALCF, and
  `container_extras/` is gitignored. The native venv + tarball route is the
  proven replacement; extend it, don't revisit containers.
- **The Midway Slurm scripts.** `scripts/train_midway*.sbatch` and
  `scripts/train.sbatch` are Slurm and reference another cluster. They are
  *templates to read*, never things to run or edit here.
- **Forcing `VLLM_USE_V1=0`.** It was tried; V0 spawns *more* startup threads
  than V1 and dies under `pids.max=4096` before training. Keep V1.
- **Editing the GREEN smoke trio in place or overwriting the proven tarball**
  (`$BASE/marshal-train-venv.tar`). New configs get new files; the extended
  venv gets a **new** tarball name. The 0.5B smoke must stay reproducible.

## Read these first (the proven artifacts)

The smoke bring-up is your template; reuse its working machinery, do not
rewrite it. The two **notes files** (`polaris_pbs_notes.md`,
`polaris_pbs_setup_guide.md`) are the deliberate exception — you will *revise*
them per **Documentation discipline**, not preserve them verbatim.

| File | What it is | Use it for |
|---|---|---|
| `polaris_pbs_notes.md` | Lab notebook: cluster facts, the 10-layer fix stack, full failure log | Ground truth + the file you extend |
| `polaris_pbs_setup_guide.md` | Step-by-step recipe; its final "Scaling to a more complex LLM" section is this mission's outline | The procedure you scale up |
| `polaris_scaleup_gpu_needs.md` | Plain-language scale-up explainer (written *before* this work) | Background; update it to realized tense (Objective 5) |
| `midway_notes.md` — "GREEN — megatron_train path passed" (jid 50261211) | The proven megatron run: config, fixes (`NVTE_FUSED_ATTN=0`, `pipefail`, `CUDA_DEVICE_MAX_CONNECTIONS=1`), artifact shapes | The template result you reproduce on Polaris |
| `examples/tictactoe/agentic_val_tictactoe_selfplay_midway_megatron.yaml` + `run_..._midway_megatron.sh` + `scripts/train_midway_megatron.sbatch` | The Midway megatron trio | Source for the Polaris copies |
| `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_smoke.yaml` + `run_..._polaris.sh` + `scripts/train_polaris.pbs` | The GREEN Polaris trio | The Polaris-specific machinery (tarball staging, thread caps, probes) you inherit |
| `scripts/polaris_limits_probe.pbs` | The cgroup-limits diagnostic (measured `pids.max=4096`) | Re-run/extend it to measure any new layout's pids headroom |
| `scripts/polaris_gpu_probe.py` | Granular, unbuffered GPU/torch probe | Keep in the wrap; extend for megatron imports if useful |
| `mcore_adapter/` | The local Megatron-Core adapter package (`pip install ./mcore_adapter`) | Read its `setup.py`/requirements to resolve the megatron-core version (Objective 1) |
| `docker/Dockerfile.torch260.vllm` | The recipe behind the Midway container | The toolchain pins + apex/flash-attn install incantations to mirror |
| `roll/distributed/strategy/vllm_strategy.py` | Carries the `RecvBucketManager` try/except stub | With the real `mcore_adapter` installed the real import must win — verify (Objective 1) |
| `roll/third_party/vllm/vllm_0_8_4/llm.py` + `roll/distributed/executor/worker.py` + `roll/pipeline/agentic/agentic_pipeline.py` | The three in-repo source patches (V1 `.cpu().float()` defense; air-gap node-IP fix; val-scheduler skip) | Imported from the repo via `PYTHONPATH`, so they apply with no tarball rebuild — do not regress them |
| `../decrypto/polaris_pbs_notes.md` | Sibling port's notebook — re-verified queue limits (`qstat -Qf`, 2026-06-12), confirmed A100 40 GB (no 80 GB partition), staged + vLLM-probed Qwen3-4B (job 7197265) | Cross-check cluster facts; confirm the relative path resolves before citing it |

## Confirmed cluster facts (re-verify the ones marked ⟳)

| Item | Value | Note |
|---|---|---|
| Scheduler | PBS Pro — `qsub` / `qstat` / `qdel` | bare job id: `${PBS_JOBID%%.*}` |
| Account (`-A`) | `lighthouse-uchicago` | NOT "Uchicago-lighthouse" — PBS rejects that. Allocation 12374, ~17,000 node-h available (`sbank-list-allocations`) |
| Filesystems | `-l filesystems=home:eagle` | REQUIRED on every job |
| Select line | `-l select=N:ncpus=64:ngpus=4` | do NOT add `:system=polaris` |
| GPU | 4× NVIDIA A100-SXM4-**40 GB** (sm_80, NVLink) per node | re-confirmed 2026-06-12 on-node (`nvidia-smi`, decrypto job 7197265) + the [compute-nodes doc](https://docs.alcf.anl.gov/polaris/#polaris-compute-nodes); there is no 80 GB partition |
| CPU / RAM | EPYC Milan 7543P, 32c/64t (`ncpus=64`), 512 GiB | |
| **Job cgroup `pids.max`** | **4096** threads+processes for the whole job | measured, jid 7186710 (`scripts/polaris_limits_probe.pbs`); root-owned, cannot be raised from inside a job |
| Native CUDA | 12.4.1 (`/soft/compilers/cudatoolkit/cuda-12.4.1`, nvcc 12.4.131) | exact match to the venv's torch cu124 |
| Conda | `module use /soft/modulefiles && module load conda/2025-09-25 && conda activate base` | then activate the venv manually (never source a relocated venv's `activate`) |
| Member base `$BASE` | `/lus/eagle/projects/lighthouse-uchicago/members/mehta5` | repo, venv, models, caches |
| Model store | `$BASE/models/` | **`Qwen3-4B` is already staged** (7.5 GB, 3 shards, verified 2026-06-12; vLLM-loaded on an A100 in decrypto job 7197265: KV cache 189,648 tokens at TP=1). Also present: Qwen2.5-0.5B-Instruct, Qwen3-8B, Qwen2.5-72B-Instruct |
| Proven venv + tarball | `$BASE/conda-envs/marshal-train` + `$BASE/marshal-train-venv.tar` (8.3 G) | torch 2.6.0+cu124 · vllm 0.8.4 · ray 2.46.0 · deepspeed 0.16.4 · transformers 4.51.2 · py 3.12.11 |
| Queue (smoke) | `debug` — 1–2 nodes, ≤1 h, 1 running + 1 queued/user | hosted every smoke so far |
| Longer queues | `preemptable` — 1–10 nodes, ≤72 h, `max_run=[p:10]`, preemptible (`-r y`) — confirmed 2026-06-12 via `qstat -Qf`. `debug-scaling` (1–10 nodes, 1 job/user) ⟳ confirm walltime. `prod` forces ≥10 nodes — not for us | re-confirm before the 20-step run; limits change |
| Tracking | tensorboard (compute nodes are air-gapped; wandb would need offline mode + later sync) | |

**ALCF Polaris documentation — consult it when you need authoritative cluster
detail**, and use it to *confirm* the ⟳ rows rather than trusting this prompt:

- Compute-node hardware: <https://docs.alcf.anl.gov/polaris/#polaris-compute-nodes>
- Running jobs / queue table: <https://docs.alcf.anl.gov/polaris/running-jobs/>
- Docs landing page (modules, software, multi-node NCCL guidance): <https://docs.alcf.anl.gov/polaris/>

You have web access; fetch these when a fact is marked ⟳ or this prompt does
not cover something. When the docs and this prompt disagree, the docs win —
record the correction in the notes.

**Carry-over gotchas that still apply** (all already coded into the proven
Polaris trio — preserve them): venv must be staged from the tarball to
node-local SSD and activated manually (eagle's small-file latency hangs
`import torch` otherwise — measured 71,700 files = 1137 s vs one 8.3 G tarball
= ~30 s); compute nodes are **air-gapped** (`HF_HUB_OFFLINE=1`, local model
paths only, no `8.8.8.8` tricks — the node-IP source patch handles this); CPU
thread caps `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB/RAYON=1` + `RAY_NUM_CPUS=16` +
`MAX_JOBS=4` exported *before* python (pids headroom); `TMPDIR` on node-local
scratch; `CUDA_HOME=cuda-12.4.1` + `CC=gcc-12`/`CXX=g++-12` (nvcc 12.4 rejects
the conda module's gcc-14); persistent `TORCH_EXTENSIONS_DIR=$BASE/torch_extensions`
+ `TORCH_CUDA_ARCH_LIST=8.0`; `device_mapping` values must be **strings** (ROLL
`eval()`s them — a bare YAML list dies at config parse); `set -o pipefail` in
every launcher (a `python | tee` tail otherwise masks the real exit code);
watch `logs/wrap_<jid>.log` live, not the PBS `.OU`/`.ER` spool (flushes only at
job end); check `pbsnodes -l` before blaming your setup (debug nodes go down in
batches with prologue/Lustre faults — a full-hour 0-byte run is *their* outage);
and the **startup EAGAIN race**: the import "thundering herd" peaks near
`pids.max=4096` and killed ~half the smoke's submissions at
`RolloutScheduler.__init__` — a lost run *hangs*, so `qdel` and resubmit rather
than debug it.

---

## Objective 1 — Extend the venv with the megatron stack (login node, one-time)

The smoke deliberately deferred `megatron-core`, `transformer-engine`, `apex`,
`flash-attn`, and `mcore_adapter` — `deepspeed_train` + eager attention needs
none of them. `megatron_train` needs all of them. Install on the **login node**
(outbound network + nvcc; compilation needs no GPU), into the existing
`$BASE/conda-envs/marshal-train` venv.

**Version targets — match what the Midway container actually shipped** (verified
by `apptainer exec` there, recorded in `midway_notes.md`): `megatron.core
0.12.3`, `mcore_adapter 0.6.0.dev0` (the repo's own package), `transformer_engine
2.2.0`, `flash_attn 2.7.2`, apex from `git+https://github.com/NVIDIA/apex.git@25.04`.
`docker/Dockerfile.torch260.vllm` is the closest written recipe but pins
`megatron-core==0.11.0` — a known discrepancy with the container's 0.12.3.
Resolve it by reading `mcore_adapter/setup.py` (the Midway notes record that
installing `./mcore_adapter` pulls megatron-core 0.12); prefer the
container-verified set, and record what you actually resolved.

Build mechanics and traps:

- **Guard the proven pins.** After *every* install step, re-verify
  `torch==2.6.0+cu124`, `transformers==4.51.2`, `tokenizers<0.22`, `numpy<2.0`,
  `vllm==0.8.4`, `ray==2.46.0`, `deepspeed==0.16.4`. The bring-up already saw
  pip drag in transformers 5.x / numpy 2.x once — if a megatron dep bumps them,
  re-pin immediately. Use `--no-build-isolation` for the source builds so they
  compile against the venv's real torch instead of a hidden fresh one.
- **flash-attn:** the Dockerfile installs the prebuilt
  `flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310` wheel. Our venv is
  **python 3.12**, so you need the **cp312** variant of the same release (it
  exists on the GitHub releases page); confirm the ABI flag matches
  `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` before
  picking the wheel. A from-source build is the fallback — hours, bound it with
  `MAX_JOBS`.
- **transformer-engine 2.2.0:** pip builds it from source against
  `CUDA_HOME=/soft/compilers/cudatoolkit/cuda-12.4.1` and needs **cuDNN**
  headers/libs. Find cuDNN on the cluster first (`module avail cudnn`, or under
  `/soft`) — do not start the build and discover the missing dep 40 minutes in.
  `CC=gcc-12`/`CXX=g++-12` here too (nvcc 12.4 caps the host compiler at 13.2).
- **apex:** mirror the Dockerfile invocation
  (`pip install --no-build-isolation --config-settings "--build-option=--cpp_ext --cuda_ext ..." git+...@25.04`).
  Long compile; `MAX_JOBS=4`-bound; `TORCH_CUDA_ARCH_LIST=8.0` keeps it to sm_80.
- **`pip install ./mcore_adapter`** last, so its megatron-core resolution lands
  on the version you chose. After it, the real `RecvBucketManager` import in
  `roll/distributed/strategy/vllm_strategy.py` must win over the stub —
  verify `python -c "from mcore_adapter.models.converter.convert_utils import RecvBucketManager"`
  succeeds and that the stub's `try` path now imports the real module. **Keep
  the stub in place** (it is what lets the deepspeed-only smoke run without
  megatron); just confirm it is dormant.
- **Rebuild the tarball under a NEW name** — e.g.
  `$BASE/marshal-train-megatron-venv.tar` — and leave
  `$BASE/marshal-train-venv.tar` untouched as the proven-GREEN fallback. The
  wrap already takes the tarball via `qsub -v MARSHAL_VENV_TARBALL=...`, so no
  script change is needed to select it. Expect the tarball to grow well past
  8.3 G; check eagle quota before packing (decrypto measured ~1.3 TB free on
  2026-06-12 — re-check).

**Login-node validation before any qsub:** import the full new surface
(`megatron.core`, `mcore_adapter`, `transformer_engine`, `flash_attn`, `apex`)
*and* re-run the proven import set + the hydra `compose()` dry-run of the new
config (Objective 3). flash-attn's CUDA kernels can only run on a GPU, so a
login-node import proves linkage, not execution — the first debug job covers
that (validation ladder rung 2).

## Objective 2 — Figure out the placement: memory × pids (the core problem)

Do not guess the layout — derive it, then **verify each candidate on the
cluster** and record the evidence (job id + the log line that proves it).

**The two hard constraints**

- **Memory (40 GiB/GPU).** Qwen3-4B ≈ 4.0 B params → bf16 weights ≈ 8 GB. With
  the Midway megatron config's settings (bf16 weights, fp32 grad accumulation,
  fp32 master weights + Adam moments), training state is ≈ **18 bytes/param ≈
  72 GB**, sharded across the TP ranks. Starting hypotheses on 40 GB cards:

  | actor_train layout | ≈ train state/GPU | Verdict to test |
  |---|---|---|
  | TP=4 (one full node) | ~18 GB | fits with room for activations (recompute=full); but see pids below |
  | TP=2 | ~36 GB | at the edge — likely no headroom for activations/CUDA context; treat as doubtful until measured |

  Caveats to check rather than trust: with DP=1 the `use_distributed_optimizer`
  flag has no extra ranks to shard over (its savings may be nil here — read the
  megatron/mcore_adapter docs or code); and the vLLM `actor_infer` copy (8 GB
  weights + KV slice set by `gpu_memory_utilization`) and the `hf_infer`
  reference (8 GB) need homes too. ROLL's colocate design parks roles between
  phases (`model_update_end_onload/offload` in the GREEN logs), which is what
  let Midway run all three roles on the same 4 GPUs — but Midway had 140 GB
  cards. On 40 GB cards a colocated layout needs the vLLM
  `gpu_memory_utilization` cut hard (the Midway megatron config's 0.6 means
  24 GB on an A100 — too much; the 0.5B smoke used 0.3) and the offload
  behavior *verified*, not assumed.
- **Head divisibility.** TP must divide both `num_attention_heads` and
  `num_key_value_heads`. Read them from
  `$BASE/models/Qwen3-4B/config.json` — do not hardcode from memory. (The
  Midway notes record 32 attention heads; expect 8 KV heads; both admit TP=4
  and TP=2. Qwen2.5-0.5B's 14 heads are why the megatron path *requires* the
  4B model.)
- **pids (`pids.max=4096` for the whole job).** Measured data points from the
  bring-up: **3 GPU workers** (1/role) + the full headroom-knob stack passed
  (with a residual ~50% startup race); **~9–12 GPU workers** failed with EAGAIN
  every time, even with thread caps. Every additional Ray GPU worker costs
  hundreds of threads (vLLM/DeepSpeed/NCCL/gRPC, not governed by OMP caps).

**Candidate layouts** (each `device_mapping` written as a string in the YAML):

| | actor_train | actor_infer | reference | GPU workers | pids prognosis | memory prognosis |
|---|---|---|---|---|---|---|
| A. Midway mirror | TP=4 `"[0,1,2,3]"` | `"[0,1,2,3]"` | `"[0,1,2,3]"` | 12 | over the cap (measured shape) | proven on H200s only |
| B. TP=4 + single-GPU infer/ref | TP=4 `"[0,1,2,3]"` | `"[0]"` | `"[1]"` | 6 | between the 3-pass and 12-fail data points — measure | train shares GPUs 0/1 with infer/ref — needs low vLLM util + verified offload turn-taking |
| C. TP=2 on distinct GPUs | TP=2 `"[0,1]"` | `"[2]"` | `"[3]"` | 4 | near the proven smoke shape — likely fits | ~36 GB/rank train state — likely too tight; measure before trusting |

Work the problem in this order: file the ALCF ticket (below), then try **B**,
falling back to **C with memory dials** (smaller `sequence_length`, vLLM util,
batch sizes) or **A if/when the pids cap is raised**. Whatever you pick, run a
**pids census** on the live job (sample
`/sys/fs/cgroup/jobs/<jid>/pids.current` from the wrap in a background loop —
extend the limits-probe pattern) so the ledger records measured headroom, not
vibes.

**File the ALCF ticket early.** The durable fix for the whole pids class —
including the smoke's residual 50% startup race — is ALCF raising the per-job
cgroup `pids.max` on the GPU nodes. It is their setting (root-owned); cite the
measurement (jid 7186710, `scripts/polaris_limits_probe.pbs`, clone failure at
4093 threads) and the workload (Ray + vLLM + DeepSpeed/Megatron multi-role RL).
File it on day one and proceed with layout engineering in parallel — do not
block on the ticket.

**Weight-sync is a different mechanism here — verify it, don't assume.** The
deepspeed smoke's per-parameter sync needed the distinct-GPU NCCL-broadcast
workaround (vLLM 0.8.4 V1 cannot msgpack a tensor). The megatron→vLLM path
instead goes through the **bucket** path (`update_parameter_in_bucket` →
`RecvBucketManager.process_bucket` — exactly what the stub stubs out today),
which carries its own V1 handling. Read `model_update_group.py:make_comm_plan`
and `megatron_strategy`'s `model_update` before the first run; treat the first
`model_update` of the smoke as the highest-risk moment, and capture the
`weight update progress: 100%` line as its proof. The `.cpu().float()` defense
in `vllm_0_8_4/llm.py` stays regardless.

## Objective 3 — Port the config trio (new files, Polaris deltas)

Author three new files; do not edit the Midway megatron trio or the GREEN
Polaris smoke trio in place:

- `examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_megatron.yaml`
- `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_polaris_megatron.sh`
- `scripts/train_polaris_megatron.pbs`

Start each from its proven parent (Midway megatron yaml × Polaris smoke
launcher/wrap) and apply, at minimum:

**YAML** (vs `agentic_val_tictactoe_selfplay_midway_megatron.yaml`):
- `exp_name` → something fresh (e.g. `agentic_tictactoe_selfplay_polaris_megatron`)
  so results land in their own `results/.../` tree.
- `pretrain` → `$BASE/models/Qwen3-4B` (the eagle store; already staged).
- `device_mapping` per the Objective 2 layout, **as strings**.
- `tensor_model_parallel_size` per the layout (the Midway file says 4).
- vLLM `gpu_memory_utilization` 0.6 → the value your memory math supports.
- `env_groups: 16` → **2** (+ `n_groups: [2]`) for both env managers — the
  measured pids fix; megatron makes pids *tighter*, never looser.
- Keep: 3 steps, tensorboard tracking, `eval_steps: 100 > max_steps` (the
  in-repo `agentic_pipeline.py` patch then skips the val scheduler — fewer
  startup actors), `template: qwen3`, `max_new_tokens: 1024` (Qwen3 thinks in
  `<think>` blocks; starving it of tokens produces format-retry failures that
  masquerade as pipeline bugs — a model behavior, not a fault; the decrypto
  notes document the same).

**Launcher** (vs `run_..._polaris.sh`, merging the Midway megatron launcher's
additions): keep the Ray cleanup preamble, thread caps (=1),
`RAY_NUM_CPUS=16`, `VLLM_USE_V1=1`, `MAX_JOBS=4`, `set -o pipefail`; add
**`CUDA_DEVICE_MAX_CONNECTIONS=1`** (megatron TP+sequence_parallel
work-ordering requirement — ROLL does not set it) and **`NVTE_FUSED_ATTN=0` +
`NVTE_FLASH_ATTN=1`** (the Midway fix: TE's cuDNN fused-attention backend had
no execution plan for Qwen3's GQA + sequence_parallel graph; flash-attn's own
kernels worked. Different cuDNN here, so it *might* work without — but the
flags are proven and cheap; keep them, note them, revisit only out of
curiosity).

**Wrap** (vs `train_polaris.pbs`): inherit everything (tarball staging, manual
venv activation, TMPDIR, caches, gcc-12/CUDA_HOME, GPU probe, live
`logs/wrap_<jid>.log` tee); default `MARSHAL_VENV_TARBALL` to the **new**
megatron tarball; point at the new launcher. Decide whether `mcore_adapter`
resolves from the venv (pip-installed, in the tarball) or from
`mcore_adapter/src` on `PYTHONPATH` (repo, like Midway did in-container) — the
venv route is simpler since the package is now installed; pick one and record it.

Validate with a login-node hydra `compose()` + `from_dict(AgenticConfig, ...)`
dry-run before any qsub (the bring-up's rung-1 pattern).

## Objective 4 — Queue plan: debug smokes → the 20-step proof run

- **Phase 1 — `debug`, 3-step megatron smoke.** Midway's megatron smoke took
  15m42s on 4× H200 *inside the container*; Polaris adds tarball staging
  (~1 min), slower cards, and the HF→mcore conversion at first load. It should
  still fit the 1 h cap, but watch the conversion time on the first run — if
  init alone eats half the walltime, check whether mcore_adapter caches the
  converted checkpoint and stage it ahead.
- **Phase 2 — the 20-step proof run.** Measure per-step time from the smoke and
  do the walltime math *before* picking the queue. If it fits well under 1 h,
  `debug` is fine. Otherwise: `preemptable` (1–10 nodes, ≤72 h, confirmed
  2026-06-12) with `#PBS -r y` and `resume_from_checkpoint: true` +
  `save_steps` low enough that a preemption loses little; or `debug-scaling`
  (⟳ confirm walltime/limits with `qstat -Qf` + the docs first). This is a
  single-job pattern (unlike decrypto's multi-server fan-out), so even
  1-job-per-user queues qualify.
- **Two nodes is the fallback, not the plan.** If no single-node layout fits
  memory, `-l select=2:...` doubles the GPU pool — but multi-node Ray plus
  ALCF's AWS-OFI/NCCL fabric is a genuine bring-up of its own (the ALCF docs
  warn a misconfigured fabric can hang Megatron-DeepSpeed, and the venv
  deliberately skipped that fabric). Attempt it only after exhausting
  single-node dials, and document it as its own validation rung.
- **Checkpoint hygiene.** Midway's megatron checkpoint was ~53 G (14 G/rank ×
  4 TP ranks) *per save*; the GREEN smoke's 12 G result dir is still on disk
  here. Set `save_steps` sanely, keep `ROLL_OUTPUT_DIR` on eagle (node-local
  SSD is ephemeral), and after verification prune heavy blobs keeping
  TensorBoard + logs + the checkpoint *directory listing* as proof — the
  established practice from both notes files.

## Objective 5 — Update the scale-up docs from plan to result; fix stale references

The repo carries text written *before* this work that describes it in the
future tense. A reviewer will check that every claim matches what actually ran.

- `polaris_scaleup_gpu_needs.md` — written as a forecast ("the scale-up will
  need...", "all four GPUs gang together"). Rewrite the affected parts to
  describe the layout and node count that actually worked, and correct its
  premises if your verified layout differs (e.g. if B or C won instead of the
  full-node gang).
- `polaris_pbs_setup_guide.md` — the "Scaling to a more complex LLM
  (Qwen3-4B + Megatron)" section becomes the realized procedure; the
  "Running it for a longer session" section gets whatever the 20-step run
  taught (queue, walltime, resume).
- `polaris_pbs_notes.md` — the validation ladder's "Scale / megatron — Not
  started (optional)" rung gets its real outcome.
- Confirm doc cross-links resolve (including `../decrypto/polaris_pbs_notes.md`
  references in both directions) and that nothing still points at
  Midway-only paths in Polaris-facing text.

**Searching the filesystem — important constraint.** Do **not** run `find`
over the cluster root, `$HOME`, or `/lus/eagle/...` broadly — those are large
shared Lustre trees and a wide `find` is punishingly slow and disruptive.
Scope every search to the repo and prefer fast tools: `git grep` / `rg` inside
the repo, `find` only under a specific small subdir (e.g. `find logs
-maxdepth 1`), `git ls-files` to enumerate tracked files, and `ls` on known
paths for staging checks.

## Objective 6 — Validation ladder

Climb in order; do not skip a rung. **Commit after each rung passes** (see
Working method) so a later failure reverts cleanly.

1. **[login] Toolchain extension proven offline.** Megatron-stack imports +
   proven-stack regression check + hydra compose of the new config all pass on
   the login node; new tarball packed under the new name; old tarball intact.
2. **[debug] On-GPU toolchain probe.** One cheap job staging the **new**
   tarball: the GPU probe passes, the megatron stack imports on-node, and a
   minimal flash-attn forward executes on the A100 (login nodes have no GPU,
   so this rung is the first time the new kernels actually run). Confirm the
   proven 0.5B deepspeed smoke still passes from the new tarball if anything
   in the shared env moved — the GREEN baseline must not silently rot.
3. **[debug] Megatron 3-step smoke** at the Objective 2 layout — the GREEN
   analog. Success looks like: HF→mca conversion completes, `weight update
   progress: 100%`, `pipeline step 0/1/2 finished` → `pipeline complete!`,
   a megatron-format checkpoint (`checkpoint-2/iter_0000001/mp_rank_*/
   model_optim_rng.pt` — the layout that proves the mcore_adapter save path,
   distinct from deepspeed's `bf16_zero_pp_rank_*`), and the wrap's `Training
   exited with code: 0`. Record the pids census and per-GPU memory evidence.
   *If this rung fails in a way that confounds model size with strategy*,
   insert a diagnostic rung: Qwen3-4B with `deepspeed_train` (ZeRO-3 /
   CPU-offload — the fragments are already in `examples/config/`) to isolate
   the variable, mirroring Midway's smoke → scale-up → megatron progression.
4. **[debug or the Phase-2 queue] 20-step proof run** — same config, only
   `max_steps`/`save_steps` (and queue/walltime) change, mirroring the
   Midway scale-up discipline of moving one axis at a time. Verify checkpoints
   land incrementally if on a preemptible queue.

---

## Documentation discipline (graded deliverable)

Update **`polaris_pbs_notes.md`** (lab notebook) and
**`polaris_pbs_setup_guide.md`** (recipe) as you work. Both currently use
✅/✗/⏳/GREEN-banner glyphs; **convert them and write everything new in the
register below.**

**Register — write as if a scientific reviewer and a future reader will grade it:**

- Neutral, precise, evidence-anchored prose. Every claim ties to a **job id
  and a log path**. Prefer "the megatron smoke (job <jid>) completed 3 steps in
  N min; `mp_rank_00..03` checkpoints on disk (`results/.../logs/custom_logs.log`)"
  over "megatron works".
- **No check marks or cross glyphs** (no ✅, ✔, ✗, ✘, ❌, ⏳). State outcomes in
  words — "Successful" / "Unsuccessful" / "Inconclusive" — or as a table
  column. Replace the existing `STATUS: GREEN ✅` banner and the guide's
  `✗ Didn't work:` sub-bullets with plain-language equivalents.
- Use **tables** for structured facts (cluster facts, the layout→memory→pids
  plan, the job ledger) and **bullets** for procedures and findings. Reserve
  prose for rationale.
- Record **both successful and unsuccessful approaches.** A dead end (a layout
  that blew the pids census, a TE build that missed cuDNN, an OOM at a given
  `gpu_memory_utilization`) is as valuable as a success — what you tried, what
  happened, the evidence, the resolution. Keep the dated decisions log going.

**Required artifact — the job ledger.** Maintain a table mapping **every log
file** to its job and outcome, and keep it current as you submit new jobs.
`logs/` is enumerable with `ls logs/` (not `git ls-files`, and never a
cluster-wide `find`). Every Polaris jid already has `logs/<jid>.OU/.ER`, most
have `logs/wrap_<jid>.log` and `logs/nvidia-smi_<jid>.txt`, and every one is
already narrated in the notes' decisions log — seed the ledger from that
narration. (The `train_midway_*` / `pull_*` logs in the same directory belong
to `midway_notes.md`'s ledger, not this one.) Exemplar rows:

| Log file(s) | Job id | Queue | What ran | Outcome | Root cause / note |
|---|---|---|---|---|---|
| `logs/wrap_7186746.log`, `logs/7186746.*.OU/.ER`, `results/tictactoe_selfplay_polaris_smoke/7186746_*/logs/custom_logs.log` | 7186746 | debug | 0.5B deepspeed smoke, roles on GPUs 0/1/2, val disabled | Successful | The GREEN bring-up: 3 steps, `pipeline complete!`, 12 G `checkpoint-2`, `Training exited with code: 0` |
| `logs/limits_probe.out` | 7186710 | debug | cgroup limits probe | Successful | Measured the governing constraint: `pids.max=4096`; live clone test failed at 4093 threads |
| `logs/7185571.*.OU/.ER` (no wrap log — predates the live tee) | 7185571 | debug | smoke attempt 1 | Unsuccessful | Bad node `x3004c0s25b0n0`: wedged D-state, zero cput, unkillable via qdel |
| `logs/wrap_7186697.log`, `logs/nvidia-smi_7186697.txt`, `logs/7186697.*.OU/.ER` | 7186697 | debug | first run with tarball→SSD staging | Unsuccessful overall, but proved the staging fix (8.4 G extracted in 12 s; `import torch` 1.5 s); died at ROLL's `get_node_ip()` dialing 8.8.8.8 on an air-gapped node |
| `logs/wrap_7186711.log`, `logs/7186711.*.OU/.ER` | 7186711 | debug | 1-GPU colocated smoke | Unsuccessful | Cleared all infra; died at first `model_update` — vLLM V1 msgpack cannot serialize a CUDA tensor (the finding that led to distinct-GPU NCCL broadcast) |

Complete the remaining rows (7185610, 7185629, 7185641, 7185650, 7185659,
7185664, 7185695, 7186701, 7186704, 7186707, 7186716, 7186717, 7186720,
7186727, 7186732, 7186739, 7186742) from the decisions log, then add a row for
every job you submit (the toolchain probe, each layout attempt, the smoke, the
20-step run), classified Successful/Unsuccessful with the one-line root cause
anchored to the log line that proves it.

## Working method

- **Branch.** You are on `polaris-pbs-handoff`. Stay on it (or branch from it).
- **Commit after every successful rung.** As soon as a rung passes (the
  tarball validates, the probe runs, the smoke closes), `git add` the changed
  scripts/config/notes and commit with a message naming the job id and what it
  proved (e.g. `megatron 3-step smoke GREEN, job 72xxxxx; layout B, pids peak 3100`).
  - **On a failure, revert to the last good commit** rather than debugging on
    top of a broken tree (`git checkout -- <file>`, or `git reset --hard
    <last-green-sha>` if badly diverged) — the ledger and the git history
    should agree on what is green. Never let an unproven change sit
    uncommitted on top of a proven one.
  - Do not `git push` or open PRs unless the user asks.
- **Distinguish your bugs from ALCF's infrastructure.** The bring-up lost five
  jobs to bad nodes and a cluster-wide prologue/Lustre outage. Before debugging
  a hang: check `logs/wrap_<jid>.log` got created at all (no log = the script
  never ran = their prologue), check `pbsnodes -l`, and check whether the same
  config already passed on another node. Resubmitting into an outage is futile;
  note it and wait.
- **Retry the startup race, don't debug it.** Until the ALCF ticket lands, a
  job that hangs at `RolloutScheduler.__init__` with EAGAIN lost the pids
  race — `qdel`, resubmit, and record the attempt in the ledger.
- **Never `find` the cluster.** Repo-scoped `git grep`/`rg`, depth-limited
  `find` under small subdirs only, `ls` on known paths.
- **Watch live, not the PBS spool.** `.OU`/`.ER` flush only at job end; tail
  `logs/wrap_<jid>.log` and `results/.../logs/custom_logs.log`.
- **Prefer `*_polaris_megatron` copies over editing proven files in place** so
  every port stays diffable.
- When a cluster fact contradicts this prompt, trust the cluster and record
  the correction in the notes.

## Definition of done

- The megatron venv extension is built, tarballed under a new name, and proven
  on a GPU node; the original tarball and the 0.5B GREEN smoke remain intact
  and reproducible.
- A **megatron 3-step smoke is GREEN on Polaris**: `pipeline complete!`,
  megatron-format `mp_rank_*` checkpoints, `Training exited with code: 0`,
  at a layout whose memory fit and pids headroom are *measured* (census in the
  ledger), with the jid + log paths recorded.
- The **20-step proof run completes** on an appropriate queue (walltime math
  shown), with checkpoints written incrementally if the queue preempts.
- The ALCF `pids.max` ticket is filed and its number + status recorded in the
  notes (resolution is ALCF's timeline, not a blocker for done).
- `polaris_pbs_notes.md` + `polaris_pbs_setup_guide.md` are updated to the
  register above: no check-mark glyphs, both successful and unsuccessful
  approaches recorded, and a complete **job ledger** covering every Polaris
  log file; `polaris_scaleup_gpu_needs.md` rewritten from forecast to result.
- No stale future-tense scaling text or dead cross-links remain — verified
  with repo-scoped search, not a cluster `find`.
- Git history shows a commit at each green rung, and the tree matches the last
  green state (no unproven changes left dangling).
