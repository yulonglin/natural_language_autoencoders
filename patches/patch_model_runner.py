#!/usr/bin/env python3
"""Patch SGLang model_runner.py to fix CUDA Error 803 on Modal A100 nodes.

Root cause: Miles uses launch_server_process (HTTP server mode, subprocess.Popen),
not SGLang's Engine class (mp.Process). SGLang's built-in maybe_reindex_device_id
only wraps mp.Process.start() in the Engine code path, so it never runs here.
The scheduler subprocess inherits CUDA_VISIBLE_DEVICES=None and cudaGetDeviceCount()
tries to enumerate all 8 GPUs, failing with Error 803 under Modal's forward-compat
CUDA mode.

Fix: inside the scheduler subprocess, before any CUDA init, restrict
CUDA_VISIBLE_DEVICES to just this GPU and remap gpu_id to 0 (the remapped index
of the single visible GPU). This is the same invariant maybe_reindex_device_id
establishes from outside.
"""
import sys
from pathlib import Path

target_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/root/sglang/python/sglang/srt/model_executor/model_runner.py"
)
f = Path(target_path)

if not f.exists():
    print(f"model_runner.py not found at {target_path}, skipping")
    sys.exit(0)

code = f.read_text()
TARGET = "torch.get_device_module(self.device).set_device(self.gpu_id)"
GUARD = "CUDA_VISIBLE_DEVICES = str(self.gpu_id)"

if GUARD in code:
    print("model_runner.py already patched")
    sys.exit(0)

if TARGET not in code:
    print("target not found in model_runner.py")
    sys.exit(1)

idx = code.index(TARGET)
line_start = code.rfind("\n", 0, idx) + 1
indent = code[line_start:idx]

insert = (
    indent + "# Modal CUDA Error 803 fix: restrict to one GPU before any CUDA init.\n"
    + indent + "# When CUDA_VISIBLE_DEVICES is unset, cudaGetDeviceCount() enumerates all\n"
    + indent + "# 8 GPUs and fails with Error 803 under Modal's forward-compat CUDA mode.\n"
    + indent + "# Set CUDA_VISIBLE_DEVICES to just our GPU and remap gpu_id to 0 so\n"
    + indent + "# set_device(0) references the correct physical device.\n"
    + indent + "import os as _os\n"
    + indent + "if _os.environ.get('CUDA_VISIBLE_DEVICES') is None and self.gpu_id >= 0:\n"
    + indent + "    _os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)\n"
    + indent + "    self.gpu_id = 0  # remap: single visible GPU is always device 0\n"
)

f.write_text(code[:line_start] + insert + code[line_start:])
print("patched model_runner.py: set CUDA_VISIBLE_DEVICES before set_device()")
