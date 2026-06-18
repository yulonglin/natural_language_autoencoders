#!/usr/bin/env python3
"""Patch SGLang maybe_reindex_device_id in common.py to fix CUDA Error 803.

Root cause: Miles uses subprocess.Popen (HTTP server mode) to start SGLang.
In the HTTP server process, CUDA_VISIBLE_DEVICES=None causes torch.cuda
.is_available() to return False (Error 803 during cudaGetDeviceCount with
all 8 GPUs visible). maybe_reindex_device_id checks `not is_cuda_alike()`
and exits early WITHOUT setting CUDA_VISIBLE_DEVICES. The scheduler
subprocess (spawned via mp.Process with spawn start-method) then inherits
CUDA_VISIBLE_DEVICES=None, import-time CUDA probes see all 8 GPUs, and
cudaGetDeviceCount fails with Error 803 (sticky: all subsequent CUDA calls
also fail, including set_device()).

Fix: remove the `not is_cuda_alike()` guard. Setting an OS env var before
mp.Process.start() doesn't require CUDA to be available — it's a pure Python
os.environ write. With the guard removed, maybe_reindex_device_id sets
CUDA_VISIBLE_DEVICES=str(gpu_id) (e.g. "4") before the scheduler subprocess
starts, the spawn inherits it, and import-time probes see exactly 1 GPU.
"""
import sys
from pathlib import Path

target_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/root/sglang/python/sglang/srt/utils/common.py"
)
f = Path(target_path)

if not f.exists():
    print(f"common.py not found at {target_path}, skipping")
    sys.exit(0)

code = f.read_text()

GUARD = "# Modal Error 803 fix: removed is_cuda_alike() guard from maybe_reindex_device_id"
if GUARD in code:
    print("common.py already patched")
    sys.exit(0)

OLD = (
    "if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() is False"
    " or not is_cuda_alike():"
)
NEW = (
    "if envs.SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS.get() is False:"
    "  " + GUARD
)

if OLD not in code:
    print(f"target not found in common.py — SGLang version mismatch?")
    print(f"Looking for: {OLD!r}")
    sys.exit(1)

f.write_text(code.replace(OLD, NEW, 1))
print(
    "patched common.py: removed is_cuda_alike() guard from maybe_reindex_device_id"
    " so CUDA_VISIBLE_DEVICES is always set before scheduler subprocess spawn"
)
