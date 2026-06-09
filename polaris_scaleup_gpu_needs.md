# Scaling up to the Megatron training run on Polaris — what it will take (plain-language note)

This note explains, in everyday terms, how much GPU horsepower we'll need to go from the
small test we just got working to the real "Megatron" training run, and where the real
difficulty is. Companion to `polaris_pbs_notes.md` (which has the technical detail).

---

## The one-paragraph answer

The test we just passed used a **small** model and gave each job-role its own GPU. The
scale-up uses a model that is **about eight times bigger**, and that bigger model does not
fit on a single GPU. So instead of one GPU per role, **all four GPUs on a Polaris node have
to gang together and act as one big GPU** to hold and train the bigger model. In short:
the scale-up needs **a whole Polaris node (its 4 GPUs), all working together**, rather than
the 3-separate-GPUs arrangement the test used. The harder problem is **not** raw GPU power —
it's a per-job "how many things can run at once" limit on Polaris that the bigger setup
bumps into. That limit, not the GPUs, is what we'll have to get raised.

---

## What's actually changing

| | The test run (already GREEN) | The Megatron scale-up |
|---|---|---|
| Model size | ~half a billion adjustable values (Qwen2.5-0.5B) | ~four billion (Qwen3-4B) — roughly **8× bigger** |
| How the model sits on GPUs | small enough to put a full copy on one GPU | too big for one GPU — must be **split across all 4 GPUs that work as a team** |
| GPUs used | 3 of the 4 (one per role, kept separate) | **all 4 of a node, ganged together** |
| Training engine | the lighter "DeepSpeed" engine | the heavier-duty **Megatron** engine, built for big models |
| Training length | 3 steps (just to prove it runs) | longer real runs |

The "split the model across 4 GPUs that work as a team" idea is the heart of it. When a model
is too big to fit on one card, you cut it into four pieces, put one piece on each GPU, and the
four GPUs constantly talk to each other to do one model's worth of work. Megatron is the tool
that does this cutting-and-coordinating.

---

## How many GPUs, concretely

- **Minimum target: one full Polaris node = 4 A100 GPUs.** Each Polaris GPU has 40 GB of
  memory; a node's four add up to 160 GB. The 4-billion model plus everything training needs
  (a working copy of the model, the "how to adjust it" bookkeeping, and a separate copy used
  for generating game moves and another for comparison) is meant to be spread across those
  four GPUs.

- **One node is the plan, but memory will be tight.** Here's the catch: the existing scale-up
  recipe was written for Midway's GPUs, which have **about 3.5× more memory each** (140 GB vs
  Polaris's 40 GB). On Polaris's smaller GPUs we are packing the same big model into much less
  room. The setup avoids running out of memory by having the roles **take turns** on the GPUs
  (one role's data is temporarily parked aside while another runs), so they don't all need
  room at the same instant. That trick should make 4 billion fit on one node — but it is close
  to the edge, so the first scale-up attempts will likely need small dials turned down
  (smaller batches of work at a time, or parking more data in regular computer memory).

- **If one node turns out to be too tight: two nodes (8 GPUs).** That doubles the memory to
  work with and removes the squeeze, at the cost of asking the scheduler for two machines and
  the extra coordination of GPUs talking across machines. This is the fallback, not the
  starting point.

**Bottom line on hardware:** plan for **one Polaris node (4 GPUs) to start**, and be ready to
ask for **two nodes (8 GPUs)** if memory proves too tight. We have plenty of allocation for
this (the account has thousands of node-hours available).

---

## The real bottleneck is NOT the GPUs

This is the important part. Getting the small test to run was hard not because we ran out of
GPU power, but because **Polaris caps how many separate programs and helper threads a single
job is allowed to run at once** (the cap is 4,096). Our training setup launches a lot of small
helper programs, and the small test only fit under that cap once we trimmed it down to three
GPUs with one program each.

The Megatron scale-up makes this **worse**, because ganging 4 GPUs together for each of the
three roles means roughly **four times as many helper programs** as the test used. That will
sail past the 4,096 cap. So before the scale-up can run reliably, **one of these has to
happen:**

1. **Ask ALCF to raise that per-job limit** on the GPU nodes (the clean fix — it's their
   setting, not something we can change ourselves). This is the recommended path and the
   single thing most likely to unblock the scale-up.
2. **Or trim the helper-program count further** (fewer parallel game environments, leaner
   settings) so the bigger run still squeezes under 4,096 — possible, but fiddly and limiting.

In other words: **the GPUs are not the constraint — the job's "number of running pieces" limit
is.** Money/allocation is fine; raw compute is fine; we mainly need ALCF to lift that ceiling
(or we keep the run deliberately small).

---

## Suggested order of operations for the scale-up

1. **File the ALCF request to raise the per-job program/thread limit** (4,096 → higher) on the
   GPU/debug nodes. Everything else waits on this.
2. Once raised, run the 4-billion model on **one node (4 GPUs)** with the Megatron engine;
   expect to lower memory-related dials on the first one or two tries.
3. If memory still won't fit on one node, move to **two nodes (8 GPUs)**.
4. Only then worry about longer training runs and any performance tuning.

---

*Numbers used: Qwen3-4B ≈ 4 billion parameters; Polaris A100 = 40 GB/GPU, 4 GPUs/node; the
existing Megatron recipe (`agentic_val_tictactoe_selfplay_midway_megatron.yaml`) splits the
model across all 4 GPUs and was sized for Midway's 140 GB H200s. The per-job limit of 4,096 was
measured on a Polaris debug node — see `polaris_pbs_notes.md`.*
