# MARSHAL Midway port — running notes

Living doc. Tracks Midway-specific bring-up of MARSHAL, following the same
playbook used for Decrypto (`/project/rcc/mehta5/decrypto/midway_notes.md`).

Scope is **cluster bring-up only**: one small smoke-config training run that
completes a handful of optimization steps and writes an artifact. NOT paper
repro, NOT eval, NOT hyperparam tuning. See `CLAUDE.md` and
`/project/rcc/mehta5/marshal_bringup_prompt.md` for the brief.

---

## Cluster facts (Midway / RCC) — same as Decrypto

| Item | Value |
|---|---|
| Account | `rcc-staff` |
| Partition | `test` only |
| GPU | H200, constraint `H200`, 4× per node, ~140 GiB VRAM each |
| Driver | 535.216.03 (max CUDA 12.2; torch 2.8/cu128 works via Minor-Version Compatibility) |
| Python module | `python/miniforge-25.3.0` |
| Activation | `module load python/miniforge-25.3.0 && eval "$(mamba shell hook --shell bash)" && mamba activate <env>` |
| Project root | `/project/rcc/mehta5/MARSHAL` |
| Rollout env (existing) | `/project/rcc/mehta5/conda-envs/vllm-probe` (vllm 0.10.2, torch 2.8.0+cu128, transformers 4.55.4, python 3.12.13) |
| Training env (this port) | `/project/rcc/mehta5/conda-envs/marshal-train` (cloned from vllm-probe + MARSHAL deps) |
| Model cache | `/project/rcc/mehta5/vllm/models/` (Qwen2.5-72B-Instruct, Llama-3.1-70B-Instruct already on disk) |
| HF cache | `/project/rcc/mehta5/hf_cache` |
| Inductor cache | `/project/rcc/mehta5/torchinductor_cache` |

---

## Env strategy

Cloning `vllm-probe` → `marshal-train` so we keep the driver-535-compatible
torch 2.8.0+cu128 / vllm 0.10.2 / transformers 4.55.4 base. MARSHAL's
`requirements_torch260_vllm.txt` pins `torch==2.6.0` / `vllm==0.8.4` which
would NOT work with driver 535 (same lesson as the Decrypto port — the
shipped pins are wrong for this cluster).

Adding MARSHAL training-side deps incrementally on top, not via blanket
`pip install -r requirements_common.txt`. Tradeoff: may have to skip
`transformer-engine` and rebuild `flash-attn` from source (their wheels are
torch-ABI specific). Acceptable for smoke since TE is a perf knob, not
correctness.

---

## Env state (vllm-probe baseline, verified 2026-05-28)

```
python       3.12.13
torch        2.8.0+cu128
cuda         12.8
vllm         0.10.2
transformers 4.55.4
```

Size on disk: 11 GiB.

---

## Decisions / changes log

- 2026-05-28 — Init. Read the Decrypto playbook + MARSHAL repo layout, wrote
  CLAUDE.md, decided on **clone `vllm-probe` → `marshal-train`** rather than
  extending vllm-probe in place (don't perturb the proven Decrypto env) or
  installing fresh from `requirements_common.txt` (driver-535 incompatible
  torch/vllm pins).
- 2026-05-28 — Cloned `vllm-probe` → `marshal-train` via
  `conda create --clone /project/rcc/mehta5/conda-envs/vllm-probe --prefix /project/rcc/mehta5/conda-envs/marshal-train -y`.
  Note: `mamba create --clone` is unsupported in this miniforge build — must
  use `conda` for the clone step. Wallclock 24m15s. New env activates and
  reports identical versions to source (python 3.12.13 / torch 2.8.0+cu128 /
  vllm 0.10.2 / transformers 4.55.4). Size 11 GiB.

- 2026-05-28 — Installed MARSHAL training deps incrementally on top of the
  clone. Dropped from `requirements_common.txt`:
  * `pyext` — broken on Python 3.12 (uses removed `inspect.getargspec`).
    Only imported by `roll/utils/local_code/testing_util.py`, which is reached
    via `roll/pipeline/rlvr/rewards/code_sandbox_reward_worker.py` — i.e. the
    **rlvr** code-sandbox path. MARSHAL uses the **agentic** pipeline, so this
    import is never hit.
  * `trl==0.9.6` — not imported anywhere under `roll/` (verified via grep);
    skipping.
  * `mcore_adapter`, `megatron-core`, `transformer-engine` — only required by
    `strategy_name: megatron_train`. Smoke will use `deepspeed_train` instead.
    Imports in `roll/distributed/strategy/factory.py` are lazy.
  * `latex2sympy2*`, `antlr4-python3-runtime` — rlvr-only (math reward).
  * `flash-attn` — relying on torch 2.8 SDPA for now; revisit for perf.
  * Pinned versions like `transformers==4.51.2`, `accelerate==0.34.2`,
    `peft==0.12.0`, `datasets==3.1.0`, `numpy<2.0`, `ray<=2.46` — let pip
    resolve to versions compatible with the inherited torch 2.8 / vllm 0.10.2
    stack; document if anything bites later.

  Install batches (all exit 0):
  1. `pip install tensordict modelscope datasets peft tyro accelerate loralib jsonlines deprecated dacite codetiming more_itertools pytest isort wandb hydra-core math-verify`
     → resolved: accelerate 1.13.0, peft 0.19.1, datasets 4.8.5, tensordict
     0.12.4, wandb 0.27.0, hydra-core (already present), modelscope 1.37.1,
     etc. **Conflict warning**: this pulled setuptools 82, but vllm 0.10.2
     requires `<80`. Fixed in next batch.
  2. `pip install "setuptools>=77.0.3,<80" open_spiel gym "gymnasium[toy-text]" gym_sokoban`
     → setuptools 79.0.1, open_spiel 1.6.15 (provides `import pyspiel` — the
     MARSHAL README says `pip install pyspiel` which is **wrong**, that
     package name doesn't exist on PyPI), gym 0.26.2, gymnasium 1.3.0,
     gym_sokoban 0.0.6.
  3. `pip install deepspeed` → deepspeed 0.19.0.

  Sanity imports verified: `import pyspiel` (122 games registered),
  `import gymnasium`, `import gym`, `import gym_sokoban`, `import deepspeed`.

- 2026-05-28 — Full agentic-pipeline import surface verified:
  `from roll.pipeline.agentic.agentic_pipeline import AgenticPipeline`,
  `from roll.distributed.scheduler.initialize import init`,
  `REGISTERED_ENV_CONFIGS` = `['tictactoe', 'hanabi', 'connect_four', 'kuhn_poker', 'leduc_poker']`,
  `from roll.distributed.strategy.factory import create_strategy`. Three
  follow-up fixes were needed during this verification step:
  * `matplotlib` is imported at module load by `roll/agentic/env/tictactoe/env.py`
    (and every game env via the registry). Wasn't in `requirements_common.txt`
    but is a hard dep. `pip install matplotlib` → 3.10.9.
  * `trl` IS imported (I missed it in the earlier grep — single-file scope).
    `roll/utils/offload_states.py:7 from trl import AutoModelForCausalLMWithValueHead`
    is at module load. The `ValueHead` class was removed in trl 0.19+, so
    the pinned `trl==0.9.6` is correct in spirit but too old for modern
    transformers. Settled on `trl>=0.11,<0.19` → resolved to trl 0.18.2.
  * The initial `pip install trl` (unconstrained) **silently upgraded
    transformers to 5.9.0**, which is the exact regression the Decrypto
    lesson warns about. Refixed with `pip install "transformers<5,>=4.51"
    "tokenizers<0.22" "trl<0.19,>=0.11"` → transformers 4.55.4 / tokenizers
    0.21.4 / trl 0.18.2 / huggingface-hub 0.36.2. Anything that touches
    transformers in this env must pin `<5`.

- 2026-05-28 — Authored three Midway-flavored files for the first smoke run:
  * `examples/tictactoe/agentic_val_tictactoe_selfplay_midway_smoke.yaml` —
    smoke config. Local Qwen2.5-0.5B-Instruct, deepspeed ZeRO-2 actor (no
    megatron), tensorboard tracking, `max_steps=3`, `rollout_batch_size=16`,
    `sequence_length=4096`. Single `TicTacToe` env (no mcts opponents) to
    keep rollout cheap. vllm strategy uses `enforce_eager: true`.
  * `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_midway.sh` —
    Midway launcher; same ray-cleanup preamble as the original, honors
    `ROLL_OUTPUT_DIR` from the sbatch wrap if set.
  * `scripts/train_midway.sbatch` — `partition=test`, `account=rcc-staff`,
    `constraint=H200`, `gpus-per-node=4`, `time=02:00:00`. Drops the apptainer
    wrap and activates the `marshal-train` mamba env directly. Applies all
    inherited fixes: TMPDIR hygiene, `TORCHINDUCTOR_CACHE_DIR`, per-job
    `RAY_TMPDIR`. Keeps `--signal=B:SIGUSR1@90` + `train_autoresume.sh`
    pattern from the original sbatch.
  * Original `scripts/train.sbatch` left untouched so the diff vs. upstream
    remains visible.

- 2026-05-28 — Dry-run via hydra `compose()` + `from_dict(AgenticConfig, ...)`
  passes. Config resolves with all three model roles (`deepspeed_train` /
  `vllm` / `hf_infer`), interpolations expand, schema validates. Ready to
  submit the sbatch.

- 2026-05-28 — Smoke v1 (jid 50215500, 1m20s on midway3-0606): died on
  `init()` with
  `TypeError: LogMonitor.__init__() got an unexpected keyword argument 'gcs_publisher'`.
  Root cause: Ray API drift. Ray >=2.48 dropped `gcs_publisher=` from
  `LogMonitor.__init__` and replaced it with `gcs_client=`. MARSHAL's
  `requirements_common.txt` pins `ray<=2.46.0` (where the old API still
  worked), but **vllm 0.10.2 requires `ray[cgraph]>=2.48.0`** — the two
  pins are incompatible, no single ray version satisfies both.

  Fix: patched `roll/distributed/scheduler/log_monitor.py:LogMonitorListener`
  to try the old constructor and degrade gracefully on `TypeError` (sets
  `self.log_monitor = None` and skips the thread). Guarded the matching
  `log_monitor_thread.join(2)` in `stop()`. Cost: we lose roll's
  rank-routed log fan-out (Ray's default session logs still work). This
  is acceptable for smoke; for real runs we may want to write a proper
  shim using newer Ray's `gcs_client` API instead.

- 2026-05-28 — Smoke v2 (jid 50216085, 1m10s): log_monitor patch worked
  (warning printed, init continued, Ray cluster up on 4 H200s, placement
  group built). Crashed later at the tensorboard tracker init with
  `ModuleNotFoundError: No module named 'tensorboard'`. My miss — I chose
  tensorboard as the tracking backend but never installed it (dry-run
  doesn't instantiate trackers, so it didn't catch this). Fixed with
  `pip install tensorboard` → tensorboard 2.20.0.

- 2026-05-28 — Smoke v3 (jid 50219135, 2m04s): Ray cluster up + tracker
  init now works. Crashed in deepspeed actor_train worker with
  `FileNotFoundError: '/software/python-anaconda-2020.11-el8-x86_64/bin/nvcc'`.
  deepspeed JIT-compiles its fused ops on first use and shells out to
  `nvcc`; with no `cuda/*` module loaded, `CUDA_HOME` defaulted to the
  stale system anaconda path that has no nvcc. Fix: added
  `module load cuda/12.9` to `scripts/train_midway.sbatch` right after
  `module load python/miniforge-25.3.0`. Confirmed the module name
  against `module avail cuda` (cuda/12.9 is present), and that loading
  it puts `/software/cuda-12.9-el8-x86_64/bin/nvcc` on PATH.

  Note: Decrypto's `slurm/run_exp_midway.sbatch` deliberately doesn't
  load any cuda module — pure-inference vLLM uses precompiled kernels,
  no nvcc needed. This is the first MARSHAL-vs-Decrypto sbatch divergence.

- 2026-05-28 — Smoke v4 (jid 50219318, 7m27s): cuda/12.9 fixed nvcc, ray
  workers spawned, but deepspeed's JIT build of `fused_adam` then failed
  with `#error "You're trying to build PyTorch with a too old version
  of GCC. We need GCC 9 or later."`. System default gcc on compute is
  8.5.0. Available modules (`module avail gcc`): 4.9.0, 7.4.0, 10.2.0,
  12.2.0, 13.2.0, 15.2.0(default). Picked **gcc/12.2.0** — within
  cuda 12.x's host-compiler matrix, avoids libstdc++ ABI surprises from
  brand-new majors. Added `module load gcc/12.2.0` to the sbatch after
  `module load cuda/12.9`.

- 2026-05-28 — Audited the dockerfile (Dockerfile.torch260.vllm) and
  ROLL source against our smoke path to confirm nothing else is missing:
  * apex / transformer-engine / megatron-core — needed only by
    `strategy_name: megatron_train`; we use `deepspeed_train`. `grep
    "from apex" roll/` returns empty.
  * flash-attn — smoke uses `attn_implementation: eager`.
  * openjdk-11 — only the RLVR code-sandbox reward worker runs code in
    Java/etc.; the agentic pipeline doesn't shell out to Java. The
    `log_monitor.py` "java-worker*.log" reference is just Ray's optional
    Java workers, which we don't enable.
  * opencv-python-headless — only the visual sokoban env; we run text
    tic-tac-toe. `grep "import cv2" roll/` returns empty.
  * cuDNN system module — torch 2.8 wheel bundles cuDNN.
  Conclusion: for the agentic+deepspeed+text-env smoke, the only modules
  needed beyond Decrypto's set are cuda/12.9 + gcc/12.2.0.

- 2026-05-28 — Smoke v5 (jid 50219476, 2m06s): deepspeed JIT-build now
  succeeds (gcc/12.2.0 fix worked). Died later in the `actor_infer`
  worker with `ModuleNotFoundError: No module named 'mcore_adapter.models'`.
  Surprise: I had assumed `mcore_adapter` was megatron-only, but
  `roll/distributed/strategy/vllm_strategy.py:14` unconditionally imports
  `RecvBucketManager` from it — the megatron→vllm weight-sync helper.

  Investigation showed `RecvBucketManager.__init__` and `.clear()` (the
  only methods reached from `VllmStrategy`) don't touch megatron at all;
  the megatron dependency is purely at module-load time of
  `mcore_adapter/.../convert_utils.py` (`from megatron.core import mpu`).
  Installing the local `./mcore_adapter` would pull megatron-core 0.12,
  which would in turn want transformer-engine / apex — all of which we
  deliberately skipped.

  Fix: patched `roll/distributed/strategy/vllm_strategy.py` to wrap the
  `from mcore_adapter...` import in `try/except ImportError`, falling
  back to a tiny stub `RecvBucketManager` that only implements the two
  methods we actually call. `process_bucket()` (megatron→vllm weight
  sync) is unreachable with a deepspeed actor, so the stub is
  functionally complete for this path.

- 2026-05-28 — Smoke v6 submitted with the RecvBucketManager stub.

---

## Pivot to the official ROLL apptainer container (2026-05-29)

The source-install `marshal-train` env (smokes v1–v6) kept colliding on the
same fault line: **no single ray version satisfies both ROLL and vllm**
(ROLL's `requirements_common.txt` pins `ray<=2.46.0`; vllm 0.10.2 needs
`ray[cgraph]>=2.48.0` — see Smoke v1). Each patch (log_monitor shim,
RecvBucketManager stub, …) bought one more step but the version impasse made
the source path a game of whack-a-mole.

Decision: stop fighting the pins and run inside the **official ROLL image**,
which carries the exact version-locked stack ROLL was built against
(torch 2.6.0+cu124, vllm 0.8.4, ray 2.46.0, deepspeed 0.16.4, megatron-core,
mcore_adapter). Image staged at
`/project/rcc/mehta5/vllm/marshal_env_torch260_vllm084.sif`. New launcher
`scripts/train_midway.sbatch` wraps the per-game launcher in
`apptainer exec --nv`; the source-install-specific patches above (log_monitor,
RecvBucketManager stub) are NOT needed in the container — its ray/vllm/mcore
are all consistent. Container-specific wiring captured in the sbatch header:
per-job TMPDIR bind, `--bind /project,/scratch`, container-scoped triton +
inductor caches (`*_container` dirs), and `PYTHONPATH` carrying
`container_extras/` (pip-target for `pyspiel`) + `mcore_adapter/src` (the sif
is read-only, so we put the vendored package on the path rather than
`pip install -e`).

- 2026-05-29 — Container smoke, first attempt (jid 50247499, ~45s of Python):
  config parsed and all roles resolved inside the image, then died at
  `import transformers` → `import deepspeed` → triton, with
  `ImportError: .../triton_cache_container/<hash>/cuda_utils.so: undefined
  symbol: cuModuleGetFunction`.

  Diagnosis: `readelf -d` on the compiled `cuda_utils.so` shows **only
  `libc.so.6` as NEEDED** — Triton 3.2.0 builds it without linking
  `libcuda.so.1`, leaving `cuModuleGetFunction` (and ~20 other `cu*` driver
  symbols) UND, expecting them to resolve from a driver already loaded into
  the process's *global* symbol table. But deepspeed's triton ops run their
  `@autotune` decorator at **import time** — before torch has loaded libcuda
  globally — so the dlopen of `cuda_utils.so` finds the symbols undefined.
  The fresh cache dir (created that run, `.so` timestamped during the run)
  ruled out stale poison: the container's own Triton build is the culprit, so
  a separate `*_container` cache dir alone can't fix it.

  Inside the image, `ldconfig` resolves `libcuda.so.1` to the baked-in 470
  stub (`/usr/lib/x86_64-linux-gnu/libcuda.so.470.182.03`); the 550 compat lib
  (`/usr/local/cuda-12.4/compat/lib.real/libcuda.so.550.54.15`) is what the
  image's startup tries to `rm` and can't (read-only fs → the harmless
  `rm: cannot remove '/usr/local/cuda/compat/lib'` spam in `.err`). The
  correct driver is the 535 host libcuda that `apptainer --nv` stages into
  `/.singularity.d/libs/libcuda.so.1`.

  Fix: in `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_midway.sh`,
  `LD_PRELOAD` the host driver libcuda from `/.singularity.d/libs` right before
  the `python` call, so its symbols are in the global namespace from process
  start (globs defensively, warns if `--nv` staged nothing). Also wiped the
  poisoned `triton_cache_container/` so it recompiles clean. The compat-lib
  `rm` errors are left alone — compat is not on the load path, so they're noise.

### GREEN — MARSHAL smoke test passed (2026-05-29)

- **jid 50252477**, node midway3-0601, partition `test`, 4× H200.
- **Elapsed: 6m52s** (`sacct` State=COMPLETED, ExitCode 0:0). `Training
  exited with code: 0`.
- Config `agentic_val_tictactoe_selfplay_midway_smoke` — roles
  `actor_train: deepspeed_train` / `actor_infer: vllm` / `reference: hf_infer`,
  all on `Qwen2.5-0.5B-Instruct`, `max_steps: 3`.
- **Artifacts proving the loop closed**
  (`results/tictactoe_selfplay_midway_smoke/50252477_20260529-153229/`):
  * DeepSpeed checkpoints were written at step 2 for all 4 actor_train ranks
    + `pipeline/checkpoint-2/` (steps 0→2 completed; `checkpoint-2` is the
    last) — confirmed in `logs/train_midway_50252477.out`, e.g.
    `[torch_checkpoint_engine.py:23:save] [Torch] Saved
    .../actor_train-0/checkpoint-2/checkpoint/bf16_zero_pp_rank_0_..._optim_states.pt`.
    The heavy weight/optimizer blobs (~12G across the 4 ranks) were **pruned
    after verification** — a 0.5B smoke checkpoint has no reuse value; the
    log line + retained TensorBoard are sufficient proof.
  * TensorBoard event files under `tensorboard/` (retained, 81K) with per-step
    metrics (`response_length`, `score/mean`, `reward`, multiple `step`
    entries); `pipeline/checkpoint-2/` bookkeeping + per-rank `logs/` also kept.
- Bring-up objective met: training job starts, completes the optimization
  steps without crashing, and writes a checkpoint. Done.

### GREEN — scale-up run passed (2026-05-29)

First step beyond minimal bring-up: grow the GREEN smoke along two axes only —
model and step count — holding everything else identical to isolate variables.

- New files (kept separate so the GREEN 0.5B baseline stays untouched, per the
  repo's "author Midway equivalents, don't edit in place" convention):
  * `examples/tictactoe/agentic_val_tictactoe_selfplay_midway_scaleup.yaml`
  * `examples/tictactoe/run_agentic_pipeline_tictactoe_selfplay_midway_scaleup.sh`
    (carries the same LD_PRELOAD triton fix)
  * `scripts/train_midway_scaleup.sbatch`
- Changes vs the smoke config: `Qwen2.5-0.5B-Instruct` → **`Qwen3-4B`**
  (template `qwen2_5` → `qwen3` on all three roles), `max_steps` 3 → **20**.
  Tweaked to suit the bigger model: `max_new_tokens` 512 → 1024, vLLM
  `gpu_memory_utilization` 0.5 → 0.6. Still `deepspeed_train` ZeRO-2 — did NOT
  switch to megatron. Batch sizes / env groups unchanged from the smoke.
- Model pulled to the local store:
  `huggingface-cli download Qwen/Qwen3-4B --local-dir
  /project/rcc/mehta5/vllm/models/Qwen3-4B` (7.6G, 3 safetensor shards). Login
  node has HF reachability; ran in the `vllm-probe` env's hf CLI.
- **jid 50259767**, node midway3-0601, `test`, 4× H200. **Elapsed: 21m02s**
  (`sacct` COMPLETED, ExitCode 0:0). `Training exited with code: 0`. No OOM /
  traceback — 4B fits comfortably with all three roles colocated on the 4 H200s.
- **Artifacts** (`results/tictactoe_selfplay_midway_scaleup/50259767_20260529-225427/`):
  all 20 steps ran — `checkpoint-19` (the final step; save_steps=max_steps=20)
  written for all 4 actor_train ranks + `pipeline/checkpoint-19/`, confirmed in
  `logs/train_midway_50259767.out`. Heavy blobs (~91G across the 4 ranks)
  **pruned after verification**; TensorBoard (per-step metrics) + per-rank logs
  + `pipeline/checkpoint-19/` retained (~111M) as proof.
- Takeaway: the container path is stable beyond the 0.5B/3-step toy — a
  realistically-sized model trains end-to-end on the test partition well within
  the 2h cap. Natural next steps if continuing: checkpoint-resume / autoresume,
  or the `megatron_train` strategy (the path MARSHAL's real configs use).

### GREEN — megatron_train path passed (2026-05-30)

Validated the strategy MARSHAL's real configs actually use: `actor_train`
`deepspeed_train` → **`megatron_train`** (mcore_adapter + megatron-core), with
the upstream real config's parallelism — **TP=4, sequence_parallel, distributed
optimizer, recompute=full**. Model Qwen3-4B (megatron TP=4 needs
num_attention_heads % TP == 0; Qwen3-4B has 32 heads ✓, Qwen2.5-0.5B has 14 ✗),
3 steps. New files: `agentic_val_tictactoe_selfplay_midway_megatron.yaml` +
matching launcher + `scripts/train_midway_megatron.sbatch`.

- Container has the full megatron stack (verified via `apptainer exec`):
  megatron.core 0.12.3, mcore_adapter 0.6.0.dev0 (from the repo's
  `mcore_adapter/src` on PYTHONPATH), transformer_engine 2.2.0, apex,
  flash_attn 2.7.2.
- Added `CUDA_DEVICE_MAX_CONNECTIONS=1` to the megatron launcher (megatron
  TP+SP work-ordering requirement; ROLL doesn't set it and the upstream CMU
  launcher relied on its container env).

- **First attempt (jid 50261129, ~12 min):** megatron init + HF→mca conversion
  + model load all succeeded (TP=4 mpu, McaGPTModel, ~1.0B params/rank × 4),
  and the vLLM `actor_infer` + hf `reference` forward passes ran. It crashed at
  the **first `actor_train` forward** (`compute_log_probs`) inside
  TransformerEngine:
  `tex.fused_attn_fwd → cuDNN Error: [cudnn_frontend] No execution plans
  support the graph` (`fused_attn_f16_arbitrary_seqlen.cu`). head_dim=128 is
  standard, so this is TE's cuDNN fused-attention backend having no plan for
  Qwen3's attn graph (GQA 32/8 + sequence_parallel) on this image's cuDNN over
  the 535 / CUDA-12.2 host driver (MVC), not a dimension problem.

  Two fixes, both in the megatron launcher:
  * **`NVTE_FUSED_ATTN=0` + `NVTE_FLASH_ATTN=1`** — route TE attention through
    flash-attn (2.7.2, in the image; its own kernels, no cuDNN) instead of the
    cuDNN fused backend. This is the actual unblock.
  * **`set -o pipefail`** — the launcher ends with `python ... | tee`, so the
    exit code was tee's (0); the crashed first attempt misleadingly showed
    `sacct` COMPLETED 0:0 and "Training exited with code: 0". pipefail makes the
    launcher return python's real status. (The deepspeed launchers share this
    latent masking — harmless there only because those runs truly succeeded.)

- **GREEN (jid 50261211)**, node midway3-0606, `test`, 4× H200. **Elapsed:
  15m42s** (`sacct` COMPLETED 0:0 — now trustworthy with pipefail; 0 cuDNN
  errors). A full optimization step ran on the megatron actor: `compute_log_probs`
  + `train_step` (13.96s) executed with real metrics (`actor/pg_loss`,
  `actor/kl_loss`, `actor_train/grad_norm` 1.04, `actor/approxkl`),
  `system/step: 2`, `system/tps` ~196.
- **Artifacts** (`results/tictactoe_selfplay_midway_megatron/50261211_20260530-000111/`):
  megatron-format checkpoints `checkpoint-2/iter_0000001/mp_rank_{00..03}/model_optim_rng.pt`
  for all 4 TP ranks + `pipeline/checkpoint-2/` (distinct from the deepspeed
  `bf16_zero_pp_rank_*` layout — confirms the mcore_adapter save path). Heavy
  blobs (~53G, 14G/rank) **pruned after verification**; TensorBoard + logs +
  `pipeline/checkpoint-2/` retained (~18M).
- Takeaway: MARSHAL's real `megatron_train` path runs end-to-end in the
  container on Midway — TP=4 sequence-parallel training, mcore_adapter HF↔mca
  conversion + checkpointing all work, once TE attention is steered off cuDNN
  onto flash-attn. Both the deepspeed and megatron training strategies are now
  proven on this cluster.

---

## Retained SLURM logs (`logs/`) — map (2026-06-02)

`logs/` is `.gitignore`'d by default (it's transient SLURM scratch). We
explicitly **whitelist only the logs from runs that completed successfully**
as proof artifacts; every failed/intermediate run's `.out`/`.err` was deleted.
Each kept log corresponds to a GREEN entry above (or the container pull).

| Log file | Run | What it proves |
|---|---|---|
| `pull_50220877.{out,err}` | Container pull (jid 50220877) | The official ROLL image `marshal_env_torch260_vllm084.sif` was pulled + `apptainer inspect`'d clean. `.err` (20K) is normal apptainer pull progress on stderr, not an error. |
| `train_midway_50252477.{out,err}` | **GREEN smoke** — Qwen2.5-0.5B, deepspeed ZeRO-2, 3 steps, 6m52s | Ray cluster up, 3 optimization steps, `checkpoint-2` saved, `Training exited with code: 0`. Full untrimmed log (~311K). |
| `train_midway_50259767.{out,err}` | **GREEN scale-up** — Qwen3-4B, deepspeed ZeRO-2, 20 steps, 21m02s | **Trimmed** (1.9M → 8.5K): startup + 4×H200 inventory, Ray init, one compact metrics line per step for all 20 steps (extracted from the raw per-step JSON), `checkpoint-19` save, exit 0. The dropped bulk was 165 per-substep `memory/*` dumps + ANSI Ray worker chatter — no proof value. |
| `train_midway_50261211.{out,err}` | **GREEN megatron** — Qwen3-4B, `megatron_train` TP=4 sequence_parallel, 3 steps, 15m42s | Megatron init + HF→mca conversion, `compute_log_probs` + `train_step` with real metrics, megatron-format `checkpoint-2/iter_0000001/mp_rank_{00..03}` saved, exit 0 (trustworthy now that the launcher sets `pipefail`). Full untrimmed log (~445K). |

Deleted (failed/intermediate, narrated in the changes log above): jids
50215500, 50216085, 50219135, 50219318, 50219476, 50220249, 50244392,
50247374, 50247499 (pre-container + container-debug attempts) and 50261129
(megatron first attempt, cuDNN fused-attn crash).

Notes:
- Only the scale-up `.out` was trimmed (it was the 1.9M outlier). The smoke and
  megatron `.out`s are kept verbatim — both are <500K and small enough to carry
  as-is. If repo size matters later, the same trim recipe applies to them.
- The trim recipe lives only here, not in a script: `grep` the
  `[DRIVER ... system/step` lines, `grep -oE '\{.*\}'` to isolate the JSON,
  parse with a throwaway python that prints the headline keys (`system/step`,
  `actor/pg_loss`, `actor/kl_loss`, `actor/approxkl`, `actor_train/grad_norm`,
  `critic/score/mean`, `tokens/response_length/mean`, `system/tps`,
  `time/actor_train/train_step/total`), then prepend startup/exit/checkpoint
  lines with ANSI stripped (`sed -r 's/\x1b\[[0-9;]*m//g'`).
