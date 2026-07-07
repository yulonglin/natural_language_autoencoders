#!/usr/bin/env python3
"""Patch miles sglang_engine.py to use fork instead of spawn for the HTTP server process.

Root cause of CUDA Error 803 on Modal:
  GPU device access is an OS-level per-process privilege (cgroup device controller or
  file-descriptor) assigned by the Modal/Ray Raylet to Ray actor processes. When miles
  calls multiprocessing.Process(start_method='spawn'), spawn creates a FRESH Python
  interpreter that does NOT inherit this privilege → cuInit(0) = 803 in the scheduler.

Fix — use 'fork' so the HTTP server process inherits the Ray actor's GPU privilege:
  Ray actor (has GPU via Raylet) --fork--> HTTP server (inherits GPU) --fork--> scheduler (inherits GPU)

The HTTP server's import-time CUDA warnings (sgl-kernel/flashinfer 803) are NON-FATAL;
those libraries fall back to CPU for probing. The scheduler's torch.cuda.set_device()
is what actually needs GPU, and via the fork chain it now gets it.

Why spawn was used originally:
  CUDA init order — spawning avoids inheriting a partially-initialized CUDA context from
  the parent. On Modal that concern doesn't apply: the SGLangEngine Ray actor never touches
  CUDA (no torch imports in __init__ or before launch_server_process), so the forked child
  starts with a clean CUDA state.
"""
import sys
from pathlib import Path

TARGET_FILE = "/root/miles/miles/backends/sglang_utils/sglang_engine.py"
GUARD = "modal_fork_v1"
OLD = 'multiprocessing.set_start_method("spawn", force=True)'
NEW = ('multiprocessing.set_start_method("fork", force=True)'
       '  # modal_fork_v1: inherit Ray actor GPU cgroup/fds on Modal')

f = Path(TARGET_FILE)
if not f.exists():
    print(f"[modal_fork_v1] {TARGET_FILE} not found, skipping")
    sys.exit(0)

code = f.read_text()

if GUARD in code:
    print(f"[modal_fork_v1] already patched, skipping")
    sys.exit(0)

if OLD not in code:
    print(f"[modal_fork_v1] ERROR: target string not found in {TARGET_FILE!r}:")
    print(f"  {OLD!r}")
    # Print context around 'set_start_method' to help diagnose
    for i, line in enumerate(code.splitlines(), 1):
        if "set_start_method" in line:
            print(f"  line {i}: {line!r}")
    sys.exit(1)

code = code.replace(OLD, NEW, 1)
f.write_text(code)
print(f"[modal_fork_v1] patched {TARGET_FILE}: spawn → fork")
