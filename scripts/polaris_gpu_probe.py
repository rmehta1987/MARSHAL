#!/usr/bin/env python
"""Fail-fast, granular GPU probe for the Polaris bring-up (called by
scripts/train_polaris.pbs under `timeout`). Prints per-step elapsed timestamps,
unbuffered, so the live wrap log pinpoints exactly which torch call blocks:
import / is_available / device_count / first CUDA context (tensor on cuda).

Exit codes: 0 = GPU usable; 3 = torch.cuda.is_available() False; other = raised."""
import os
import sys
import time

T0 = time.monotonic()


def log(msg):
    print(f"[probe +{time.monotonic() - T0:6.1f}s] {msg}", flush=True)


log("python started")
import torch

log(f"imported torch {torch.__version__} (cuda build {torch.version.cuda})")
log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

log("calling torch.cuda.is_available() ...")
available = torch.cuda.is_available()
log(f"is_available={available}  device_count={torch.cuda.device_count()}")
if not available:
    log("ERROR: torch.cuda.is_available() returned False")
    sys.exit(3)

log("creating first cuda tensor (context creation) ...")
t = torch.zeros(1, device="cuda")
torch.cuda.synchronize()
log(f"CUDA COMPUTE OK on {torch.cuda.get_device_name(0)}")
