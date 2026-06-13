# The Megatron training run on Polaris — what it actually took (plain-language note)

This note explains, in everyday terms, how much GPU horsepower the real "Megatron"
training run needed on Polaris, and where the real difficulty turned out to be. It was
originally written as a forecast before the work; this version records what actually
happened (2026-06-12 for the short proofs, 2026-06-13 for the full ~10.5-hour
production run). Companion to `polaris_pbs_notes.md` (which has the technical detail,
job ids, and measurements).

---

## The one-paragraph answer

The small test used a **small** model (half a billion adjustable values) and gave each
job-role its own GPU. The real run uses a model **about eight times bigger** (Qwen3-4B)
that cannot be trained on a single GPU. It ran successfully on **one Polaris node**: the
four GPUs gang together to hold and train the big model (each GPU holds a quarter of
it), while the two helper roles — the copy that generates game moves and the frozen
comparison copy — each tuck into a corner of one of those same GPUs. As predicted, the
harder problem was **not** raw GPU power: it was Polaris's per-job cap on "how many
things can run at once" (4,096). The first green run scraped under that cap by literally
**two slots**. We then found that a single helper program had been hogging more than
half the budget (2,120 slots) for no good reason, trimmed it, and the cap stopped being
a coin flip. A request to raise the cap is still worth filing — the draft is in the
notes — but it was not needed to get the run green. And it held up under load: a full
**400-step, ~10.5-hour production run** later completed cleanly on that same single
node, surviving four automatic restarts on Polaris's preemptible queue (details below).

---

## What actually changed from the small test

| | The small test (GREEN 2026-06-06) | The Megatron run (GREEN 2026-06-12) |
|---|---|---|
| Model size | ~half a billion values (Qwen2.5-0.5B) | ~four billion (Qwen3-4B) — roughly **8× bigger** |
| How the model sits on GPUs | a full copy fits on one GPU | **split four ways** across the node's 4 GPUs, which act as one (tensor parallelism) |
| GPUs used | 3 of the 4 (one per role, kept separate) | **all 4 for training**, with the move-generator in GPU 0's spare room and the comparison copy in GPU 1's |
| Training engine | the lighter "DeepSpeed" engine | the heavier-duty **Megatron** engine, the one MARSHAL's real experiments use |
| Proof | 3 steps (job 7186746) | 3 steps (7197427) → 20-step proof (7197442) → **full 400-step / ~10.5 h production run (7198659)** |

One forecast in the earlier version of this note did **not** survive contact with
reality, in a good way: we expected every role to need all four GPUs ganged together
(twelve workers), which would have sailed far past the per-job cap and required ALCF to
raise it first. Instead, a layout with **six** workers — four for training, one each
for the two helper roles — fit both the memory and the cap, so the run went green
without waiting on ALCF.

---

## How many GPUs, concretely

- **One Polaris node = 4 A100 GPUs (40 GB each) was enough.** Measured peak memory per
  GPU during the run: 23–25 GB of 40 — comfortable, not squeezed. The bookkeeping
  needed to train the 4-billion model is about 72 GB in total, split four ways
  (~18 GB per GPU), plus the 8 GB move-generator copy on GPU 0 and the 8 GB comparison
  copy on GPU 1. The roles also take turns (parking their data when idle), which the
  run confirmed works.
- **The two-node fallback was never needed.** It remains the escape hatch if a bigger
  model ever outgrows one node, but it brings real extra complexity (GPUs talking
  across machines), so it stays a fallback.
- **Allocation cost is trivial:** the proof runs used well under one node-hour each;
  the account has ~17,000 node-hours.

---

## The real bottleneck was exactly where we predicted — with a twist

The per-job "how many programs and threads at once" cap (4,096) was indeed the hard
constraint — two of the four attempts died on it during startup, and the green run
survived it by a margin of **two**. The twist: measurement (a thread census added to
the job) showed **one single helper program was occupying 2,120 of the 4,096 slots** —
a request-dispatcher that pre-reserves room for 2,048 simultaneous conversations,
sized by its authors for fleets hundreds of times larger than ours. Telling it to
reserve 256 instead returned ~1,800 slots of breathing room and turned "scraped by
with two to spare" into a comfortable fit.

So the order of remedies ended up being:

1. **Trim what we control** (the dispatcher's reservation, idle worker pools,
   per-process thread pools) — this is what actually got the run green, and it is all
   recorded in the repo so it stays fixed.
2. **Still ask ALCF to raise the cap** — the request text is drafted in
   `polaris_pbs_notes.md`; a higher cap would remove the class of problem entirely,
   especially for bigger future configurations. It is no longer a blocker, just good
   hygiene.

---

## Going the distance — the full production run (GREEN 2026-06-13)

The 3- and 20-step runs proved the *shape* works. The real question for actual
experiments is whether it survives **hours**, not minutes — and on Polaris the only
queue that allows multi-hour single-node jobs (`preemptable`) can **kick your job off
at any moment** to make room for someone else's. So the long run had to do two things
the short runs never tested: keep training for ~10 hours, and pick itself back up every
time it got kicked off.

It did. Job 7198659 finished a **400-step run in 10 hours 49 minutes of compute**,
spread across **four automatic restarts** — exit status clean (0). What made that
work, in plain terms:

- **It saves its progress every 25 steps** (~38 minutes of work). We first tried saving
  every 50 steps and lost three restarts in a row to bad luck — each got kicked off
  before it reached a save. Halving the interval was the fix: now a restart almost
  always banks new progress before the next interruption.
- **When kicked off, the job is automatically put back in line** (the `-r y` flag), and
  on restart it **finds its most recent complete save and continues from there** —
  losing only the ~5 minutes it takes to reload, not the hours already done. The one
  subtlety that took a try to get right: a save isn't "complete" until *all four* GPUs'
  pieces are on disk *and* a tiny bookkeeping file (a hidden dotfile the training engine
  writes) is included — the restart logic now checks for exactly that before trusting a
  save.
- **Disk stayed flat** (~9.1 of 10 TB) the whole time, because we keep only the two
  newest saves plus the final one and replace the rest with a small text listing that
  proves they existed. A single save of the 4-billion model is ~8 GB; without pruning,
  fourteen of them would have piled up.

The horsepower story didn't change at all from the short runs: still one node, four
GPUs, the same comfortable 23–25 GB per GPU, and the per-job slot cap stayed a
non-issue (peak 2,384 of 4,096 — the dispatcher trim from the short runs held for the
full ten hours). The long run was a test of *endurance and recovery*, and those are now
demonstrated, not assumed.

---

*Numbers used: Qwen3-4B ≈ 4.0 billion parameters; Polaris A100 = 40 GB/GPU, 4
GPUs/node; training state ≈ 18 bytes/parameter ≈ 72 GB split across 4 GPUs; measured
per-GPU peaks 23.3/25.2/22.1/21.8 GB (job 7197427); per-job cap pids.max = 4,096
(measured, job 7186710); dispatcher thread count 2,120 (measured, job 7197427's thread
census). Production run (job 7198659): 400 steps, walltime 10:48:37, run_count 4,
Exit_status 0, ~2 min/step steady state, slot peak 2,384 of 4,096, ~8 GB per save,
disk steady ~9.1 of 10 TB. Details and the job ledger: `polaris_pbs_notes.md`.*
