# Handoff: port the MARSHAL self-play training pipeline to ALCF Polaris (PBS)

> Paste everything below the line into a fresh Claude Code session running on a
> Polaris login node, inside a checkout of this repo. It is written **to that
> Claude**. It was authored from the RCC Midway checkout (branch
> `midway-setup`), where the Slurm version of this pipeline already runs
> end-to-end (deepspeed AND megatron training paths both GREEN — see
> `midway_notes.md`).

---

You are a Claude Code session on **ALCF Polaris**. Your job is to get the
**MARSHAL self-play training pipeline** running on Polaris and prove it with a
small smoke training run on Polaris GPUs: a job that starts, completes a handful
of optimization steps without crashing, and writes a checkpoint/log proving the
loop closed. **Cluster bring-up only** — NOT paper repro, NOT eval, NOT
hyperparameter tuning. Same scope discipline as the Midway port.

Polaris uses **PBS Pro** (`qsub`/`qstat`/`qdel`), not Slurm. This same pipeline
was already ported to **UChicago RCC Midway (Slurm)**, and that port is your
template. Your task is the PBS analog of that Midway port: the scheduler wrap,
the storage paths, account/allocation/queue info, and re-confirming the
container's driver-specific fixes against Polaris's GPU driver.

## How MARSHAL differs from the Decrypto Polaris port (read this first)

If you've seen `/project/rcc/mehta5/decrypto/polaris_handoff_prompt.md`, **do
not copy its shape.** Decrypto split into a `vllm serve` **server job** plus a
separate game-loop **runner job** that discovered the server over HTTP — so most
of that handoff was about PBS server-discovery (`qstat` parsing,
`DECRYPTO_SERVERS_FILE`, job-name port encoding). **MARSHAL has none of that.**

MARSHAL is a **single training job**. One `qsub` launches one node; Ray then
spawns all three model roles as worker actors **colocated on that node's 4
GPUs**:

- `actor_train` — `deepspeed_train` (smoke) or `megatron_train` (real path)
- `actor_infer` — `vllm` strategy, for rollouts
- `reference` — `hf_infer` strategy, for the KL reference

There is **no server to discover, no job dependency, no HTTP between jobs, no
`*_SERVERS_FILE`**. Ignore everything the Decrypto handoff says about server
discovery, `qsub -W depend=`, and `ping_servers`. Your porting surface is much
smaller: it's one PBS script wrapping one container.

The second big difference: **MARSHAL runs inside an Apptainer container.** The
Midway port abandoned a source-install conda env (it hit an unresolvable
ray/vllm version impasse — `midway_notes.md` has the full chain) and switched to
the **official ROLL image**, which carries the exact version-locked stack ROLL
was built against:

```
torch 2.6.0+cu124 · vllm 0.8.4 · ray 2.46.0 · deepspeed 0.16.4 ·
megatron-core 0.12.3 · mcore_adapter 0.6.0.dev0 · transformer_engine 2.2.0 ·
flash_attn 2.7.2 · transformers 4.55.x
```

Image on Midway:
`/project/rcc/mehta5/vllm/marshal_env_torch260_vllm084.sif` (~11 GB), pulled
from `docker://roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084`.

**The container is the unit of portability.** You do NOT rebuild an env, pip
nothing, fight no version pins. You (1) get this `.sif` onto Polaris, (2) write a
PBS wrap that `apptainer exec --nv`s it, and (3) re-confirm the handful of
driver-specific env tweaks (below) against Polaris's driver. That's the whole
job.

## Read these first (the Midway template)

- **`midway_notes.md`** — the step-by-step bring-up + decisions log for the
  Slurm/container port. **Your deliverable is the PBS equivalent of this
  document.** Mirror its structure: cluster-facts table, env/container strategy,
  the GREEN entries (smoke, scale-up, megatron), retained-logs map, and a dated
  decisions log at the bottom. The "Pivot to the official ROLL apptainer
  container" section and the three GREEN entries are the most important reading —
  they tell you exactly which fixes are load-bearing.
- **`CLAUDE.md`** — the bring-up brief and repo architecture (ROLL framework
  under `roll/`, per-game configs under `examples/<game>/`, the
  `start_agentic_pipeline.py` entry point).
- **`scripts/train_midway.sbatch`** — the Slurm wrap for the deepspeed smoke.
  This is the file you're porting to PBS. Read every comment; the env-var
  passthrough block and the bind list are load-bearing.
- **`scripts/train_midway_megatron.sbatch`** — same wrap, megatron path (the
  strategy MARSHAL's real configs use). Differs only in job-name, results label,
  and the launcher it invokes.
- **`examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_midway.sh`** —
  the **in-container launcher** the wrap invokes. Ray-cleanup preamble + the
  `LD_PRELOAD` libcuda fix + the `python start_agentic_pipeline.py` call. This
  runs *inside* the image; it is mostly cluster-agnostic but contains the
  driver-specific libcuda preload you must re-confirm.
- **`examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_midway_megatron.sh`**
  — the megatron launcher; adds `CUDA_DEVICE_MAX_CONNECTIONS=1`,
  `NVTE_FUSED_ATTN=0 / NVTE_FLASH_ATTN=1`, and `set -o pipefail`.
- **`examples/tictactoe/agentic_val_tictactoe_selfplay_midway_smoke.yaml`** —
  the smoke config (Qwen2.5-0.5B-Instruct, deepspeed ZeRO-2, `max_steps: 3`,
  tensorboard). Note `pretrain:` is a **local absolute path** into the Midway
  model store — you'll repoint it to Polaris storage.
- **`scripts/pull_container_midway.sbatch`** — how the image was pulled on Midway
  (on the `build` partition, which had outbound network). The PBS analog of
  *staging* the image is your first practical step — see below.

`examples/start_agentic_pipeline.py` is the entry point and is
**scheduler-agnostic** — you should not need to touch it. Same for everything
under `roll/`.

## The Slurm→PBS porting surface (for the single training job)

| Concern | Midway (Slurm) | Polaris (PBS Pro) — what you need to do |
|---|---|---|
| Submit a job | `sbatch scripts/train_midway.sbatch` | `qsub scripts/train_polaris.pbs` |
| Directives | `#SBATCH ...` | `#PBS ...` |
| Account/allocation | `--account=rcc-staff` | `-A <project>` (your active Polaris allocation) |
| Partition/queue | `--partition=test` | `-q <queue>` (use `debug` for the smoke test) |
| Walltime | `--time=02:00:00` | `-l walltime=01:00:00` |
| Node + GPU shape | `--nodes=1 --gpus-per-node=4 --constraint=H200` | `-l select=1:ncpus=<K>:ngpus=4` (Polaris gives the whole node; confirm the `select` idiom — ALCF examples often add `:system=polaris`) |
| CPUs / RAM | `--cpus-per-task=32 --mem-per-cpu=16G` | folded into the `select` line (`ncpus=`); Polaris allocates the full node's RAM with it |
| **Filesystems** | (implicit) | `-l filesystems=home:eagle` — **Polaris rejects jobs that don't declare this**; include every FS the job touches (home + your project FS) |
| Job-local id in script | `$SLURM_JOB_ID` | `$PBS_JOBID` (looks like `1234567.polaris-pbs-...`; use `${PBS_JOBID%%.*}` for a clean numeric tag in dir names) |
| Output/error files | `--output=...%j.out` | `#PBS -o ... -e ...` (no `%j`; PBS writes `<jobname>.o<jobid>` by default, or set explicit paths) |
| List my jobs | `squeue --me` | `qstat -u $USER` / `qstat -f <jobid>` |
| Cancel | `scancel <jid>` | `qdel <jid>` |
| Pass env into job | `--export=ALL` | `qsub -v FOO=bar` for specific vars; `-V` exports your full login env (prefer setting vars *inside* the script — see the TMPDIR-hygiene note) |
| Node-local scratch | `/tmp/${USER}_${SLURM_JOB_ID}` | Polaris compute nodes have a node-local SSD (commonly `/local/scratch`) — use it for `TMPDIR`; **confirm the path on a compute node** |

Note: there is **no `--wrap`, no job dependency, and no second job** to port —
MARSHAL is one self-contained `qsub`. PBS has no `--wrap` anyway, but you don't
need it: the Midway wrap is already a real script file, so you port it
file-to-file.

## Staging the container on Polaris (your first practical step)

The `.sif` is ~11 GB. Two ways to get it onto Polaris storage; pick whichever
works:

1. **Pull it fresh** from the registry on a node with outbound network. On
   Midway that was the `build` partition; on Polaris, **login nodes typically
   have outbound network** — try the pull directly from a login node (or a
   `debug`-queue job if login-node pulls are blocked/size-limited):
   ```bash
   module load apptainer            # confirm the exact module name on Polaris
   export APPTAINER_TMPDIR=/eagle/<project>/<user>/apptainer_tmp
   export APPTAINER_CACHEDIR=/eagle/<project>/<user>/apptainer_cache
   mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
   apptainer pull marshal_env_torch260_vllm084.sif \
     docker://roll-registry.cn-hangzhou.cr.aliyuncs.com/roll/pytorch:nvcr-24.05-py3-torch260-vllm084
   ```
   ⚠️ **The registry is Aliyun (cn-hangzhou).** ALCF networks may not reach it,
   or it may be slow/blocked. If the pull stalls or fails, fall back to (2).
2. **Transfer the already-pulled `.sif` from Midway.** It exists at
   `/project/rcc/mehta5/vllm/marshal_env_torch260_vllm084.sif`. Use **Globus**
   (the ALCF-blessed path for multi-GB transfers between centers; both Midway/RCC
   and ALCF/eagle have Globus endpoints) or `scp`/`rsync` if you have direct ssh
   between the centers.

Either way, land it on your Polaris project FS (e.g.
`/eagle/<project>/<user>/marshal/marshal_env_torch260_vllm084.sif`) and verify:
```bash
apptainer inspect marshal_env_torch260_vllm084.sif
```

You also need the **model** on Polaris storage. The smoke uses
`Qwen2.5-0.5B-Instruct` (tiny — just re-download it with `huggingface-cli
download Qwen/Qwen2.5-0.5B-Instruct --local-dir <polaris-model-store>/Qwen2.5-0.5B-Instruct`
from a login node, which has HF reachability). For the scale-up/megatron path
you'd similarly stage `Qwen3-4B`. Repoint the config's `pretrain:` to wherever
you put it.

And the **`container_extras/` + `mcore_adapter/src`** that the Midway wrap puts
on the in-container `PYTHONPATH`: `container_extras/` holds the `pyspiel.so` +
`open_spiel` package built for the image (the read-only sif can't be
`pip install`ed into), and `mcore_adapter/src` is the vendored adapter. Both
live in the repo checkout, so once you `git checkout midway-setup` they're
present — just keep the same `PYTHONPATH` wiring in your PBS wrap.

## Polaris facts to CONFIRM on the cluster (don't trust these blindly)

I'm authoring this from Midway and cannot see Polaris. Verify each yourself
(`qstat -Q`, `nvidia-smi`, ALCF Polaris docs) and record them in your notes:

- **Allocation/project name** for `-A` (you need an active Polaris allocation).
- **Queue** — use `debug` for the smoke (fast turnaround, ≤1 hr). Confirm node
  count and walltime limits with `qstat -Q`.
- **`-l filesystems=` value** — almost certainly `home:eagle` (or `home:grand`).
  Jobs are rejected without it. Declare every FS the job reads/writes.
- **GPUs** — Polaris nodes are **4× NVIDIA A100 (40 GB SXM4 each)**. Confirm with
  `nvidia-smi`. **This is far tighter than Midway's H200 (~140 GB each).** The
  smoke (0.5B model, 3 roles colocated, vLLM `gpu_memory_utilization: 0.5`) fit
  comfortably on H200; on a 40 GB A100 with `actor_train` + `actor_infer` +
  `reference` all sharing the same 4 GPUs it will be tight. **Start with the 0.5B
  smoke and expect to lower `gpu_memory_utilization` and/or `sequence_length`.**
- **GPU driver version** (`nvidia-smi`) — this determines whether the container's
  `LD_PRELOAD` libcuda fix needs adjustment (see gotchas). Midway's was 535;
  Polaris's will differ.
- **Apptainer/Singularity** — confirm it's available and the exact module
  incantation (Polaris commonly: `module use /soft/modulefiles && module load
  apptainer` or a `singularity` module). Confirm `apptainer exec --nv` works and,
  critically, that **`--nv` stages the host driver `libcuda` into
  `/.singularity.d/libs/`** inside the container — the triton fix depends on it
  (see gotchas). If Polaris's container runtime stages libcuda elsewhere, adjust
  the preload path.
- **Project storage root** — `/eagle/<project>/<user>/...` or `/grand/...`.
  Repoint every Midway `/project/rcc/mehta5/...` path: repo checkout, the `.sif`,
  model store, HF cache, triton cache, inductor cache, logs, results dir.
- **Node-local scratch path** for `TMPDIR` (likely `/local/scratch`) — confirm on
  a compute node.
- **`--bind` targets** — the Midway wrap binds `/project,/scratch`. On Polaris
  bind the FS roots your job actually touches (e.g. `/eagle`, `/grand`,
  `/local/scratch`). `home` is usually auto-bound; confirm.

## Carry-over gotchas from the Midway port (these are baked into the container/launcher)

These are the fixes that made Midway GREEN. Most live in the **launcher `.sh`**
(runs inside the container) and the **wrap**, and most are driver/container
concerns that likely recur on Polaris. Re-confirm each:

- **`LD_PRELOAD` the host-driver libcuda (the big one).** Triton 3.2.0 (pulled in
  by deepspeed) builds `cuda_utils.so` with the driver symbols
  (`cuModuleGetFunction`, …) left UND, expecting them already in the global
  namespace. deepspeed triggers that dlopen during `import transformers` —
  *before* torch loads libcuda — so the import dies with `undefined symbol:
  cuModuleGetFunction` unless you preload the host libcuda. The Midway launcher
  does:
  ```bash
  HOST_LIBCUDA=$(ls /.singularity.d/libs/libcuda.so.1 2>/dev/null || ls /.singularity.d/libs/libcuda.so* | head -1)
  export LD_PRELOAD="${HOST_LIBCUDA}${LD_PRELOAD:+:$LD_PRELOAD}"
  ```
  and the wrap sets `LD_LIBRARY_PATH=/.singularity.d/libs:...` (host driver
  first) + `TRITON_LIBCUDA_PATH=/.singularity.d/libs/libcuda.so.1`. **This whole
  fix hinges on `apptainer --nv` staging the host libcuda into
  `/.singularity.d/libs/` — confirm that path on Polaris** and adjust if its
  runtime stages it elsewhere. The fix itself is driver-agnostic (it just wants
  *a* loaded driver), so it should carry over as-is if the staging path matches.
- **Container-scoped triton + inductor caches on persistent storage.** Point
  `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at per-container dirs on your
  project FS (Midway used `*_container` suffixes to keep them separate from the
  abandoned source-install caches, which had been poisoned with incompatible
  `cuda_utils.so`). A fresh empty dir on Polaris is fine — just keep it off
  transient scratch so it survives between jobs.
- **TMPDIR hygiene.** `-V`/inherited env can carry a stale `TMPDIR` from a
  cancelled job. Inside the script: `unset TMPDIR`, then
  `export TMPDIR=<node-local-scratch>/${USER}_${PBS_JOBID%%.*}`, `mkdir -p` it,
  and **bind it into the container** (`--bind "$TMPDIR:$TMPDIR"` +
  `--env TMPDIR=$TMPDIR`) so the inner process sees the same path. Put Ray's
  scratch under it (`RAY_TMPDIR="$TMPDIR/ray"`).
- **Ray cleanup preamble.** The launcher's aggressive `ray stop --force` + `pkill`
  preamble clears stale Ray sessions that poison a new head node. Keep it. The
  wrap's `cleanup()` trap (ray stop inside the image, then host-side pkill) also
  stays — it relies on Ray procs sharing the host PID ns (no `--pid` on the
  apptainer exec), which holds the same way on Polaris.
- **Do NOT load host `cuda`/`gcc` modules for the container path.** The Midway
  *source-install* attempts needed `module load cuda/12.9` + `gcc/12.2.0` to get
  `nvcc`/a modern gcc for deepspeed's JIT build — but that path was **abandoned**.
  The **container carries its own toolchain** (nvcc, gcc, the full megatron
  stack). Inside the image you need no host compiler modules; loading them risks
  shadowing the image's. Only load `apptainer` itself.
- **Megatron path only** (when you get there after the deepspeed smoke is green):
  the megatron launcher sets `CUDA_DEVICE_MAX_CONNECTIONS=1` (TP+sequence-parallel
  work-ordering) and steers TransformerEngine off its cuDNN fused-attention
  backend with `NVTE_FUSED_ATTN=0 / NVTE_FLASH_ATTN=1` (on Midway, cuDNN had "no
  execution plans support the graph" for Qwen3's GQA+SP attn over the 535/MVC
  driver; flash-attn 2.7.2 in the image sidesteps it). On Polaris's A100 +
  different driver the cuDNN path *might* work, but the flash-attn fallback is the
  safe default — keep it. Also keep `set -o pipefail` in the launcher: without it
  the final `python ... | tee` masks a crash as exit 0 and `qstat`/your notes will
  lie about success.
- **Model `num_attention_heads % TP == 0` for megatron.** TP=4 needs heads
  divisible by 4: Qwen3-4B (32 heads) ✓, Qwen2.5-0.5B (14 heads) ✗. So the
  megatron smoke must use Qwen3-4B, not the 0.5B — same as Midway.

## Deliverables

Mirror the Midway naming so the two ports sit side by side (write
PBS-flavored copies, don't edit the Midway scripts in place):

1. **`scripts/probe_container_polaris.pbs`** — single-node toolchain probe (the
   PBS analog of "does the container even run here"). `apptainer exec --nv` the
   image and confirm, inside it: `nvidia-smi` sees the 4 A100s, torch sees the
   GPUs, **`import transformers` succeeds** (this is what exercises the
   triton/libcuda fix — the Midway blocker), and a tiny vLLM load+generate of
   `Qwen2.5-0.5B-Instruct` works. **Run and pass this FIRST**, before any
   training job.
2. **`scripts/train_polaris.pbs`** — PBS analog of `train_midway.sbatch`: the
   apptainer wrap with `#PBS` directives, `-l filesystems=`, node-local-scratch
   TMPDIR + bind, caches repointed to Polaris storage, binds adjusted for the
   Polaris FS roots, and the same explicit `--env` passthrough block. Invokes the
   in-container launcher.
3. **`examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_polaris.sh`** —
   the in-container launcher (analog of the `*_midway.sh`). Likely near-identical:
   Ray cleanup, the `LD_PRELOAD` libcuda fix, `start_agentic_pipeline.py`. Adjust
   only if Polaris's container runtime stages libcuda differently.
4. **`examples/tictactoe/agentic_val_tictactoe_selfplay_polaris_smoke.yaml`** —
   smoke config analog: Qwen2.5-0.5B-Instruct (deepspeed ZeRO-2, `max_steps: 3`,
   tensorboard), `pretrain:` repointed to the Polaris model store,
   `gpu_memory_utilization` lowered for the 40 GB A100s as needed.
5. *(Optional, after the deepspeed smoke is GREEN)* the megatron trio —
   `scripts/train_polaris_megatron.pbs` +
   `run_agentic_pipeline_tictactoe_selfplay_polaris_megatron.sh` +
   `agentic_val_tictactoe_selfplay_polaris_megatron.yaml` (Qwen3-4B, TP=4,
   carrying the `CUDA_DEVICE_MAX_CONNECTIONS` / `NVTE_*` / `pipefail` knobs).
6. **`polaris_pbs_notes.md`** — the PBS equivalent of `midway_notes.md`.
   ⚠️ **Use this filename.** Mirror the Midway notes' structure (cluster-facts
   table, container-staging strategy, the validation-ladder runs, retained-logs
   map) and keep a **dated decisions/changes log at the bottom with PBS job ids**
   — that log is the single most useful artifact when something breaks.

## Validation ladder

1. **Stage the container** (pull or transfer) → `apptainer inspect` clean.
2. **Probe** — submit `probe_container_polaris.pbs`; confirm `nvidia-smi` sees
   the A100s, torch sees the GPUs, `import transformers` succeeds (triton/libcuda
   OK), vLLM loads a 0.5B model and generates. **Don't proceed until green.**
3. **Smoke (deepspeed, small)** — submit `train_polaris.pbs`; confirm: Ray
   cluster comes up on the node's 4 A100s → placement group built → 3
   optimization steps run → a `checkpoint-2` (deepspeed `bf16_zero_pp_rank_*`
   layout) is written + TensorBoard event files → `Training exited with code: 0`.
   That's the bring-up done — record it as a GREEN entry with the PBS jid +
   wallclock + the artifact path.
4. **Scale / megatron (optional)** — Qwen3-4B, then the `megatron_train` TP=4
   path with the TE flash-attn flags. These are the natural next steps Midway
   took once the smoke was green.

## Working method

- Get the code: `git fetch && git checkout midway-setup` (the Midway port + these
  notes + `container_extras/` + `mcore_adapter/src` all live on that branch; the
  remote is `git@github.com:rmehta1987/MARSHAL.git`). Consider a `polaris-setup`
  branch for your work.
- Keep a **dated decisions/changes log** at the bottom of `polaris_pbs_notes.md`
  with PBS job ids, exactly like `midway_notes.md` does — narrate each failed
  attempt and its fix, not just the GREEN ones.
- Prefer writing PBS-flavored copies (`*_polaris.*`) over editing the Midway
  scripts in place, so both ports remain diffable — the same convention Midway
  followed against the original CMU `train.sbatch`.
- When a Polaris fact contradicts an assumption in this prompt, **trust the
  cluster and note the correction** in your log.
- Done criterion (same shape as Midway's): a GREEN entry in `polaris_pbs_notes.md`
  citing the PBS jid, elapsed wallclock, and the artifact (checkpoint + log line +
  TensorBoard) that proves the training loop closed.
