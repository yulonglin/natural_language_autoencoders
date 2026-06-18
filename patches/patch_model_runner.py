#!/usr/bin/env python3
"""Patch SGLang model_runner.py to fix CUDA Error 803 on Modal A100 nodes.

Insert cuInit(0) + cudaGetLastError() before set_device() so the CUDA runtime
can recover from a sticky error state set by earlier imports (flashinfer, etc.).
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
GUARD = "cuInit(0)"

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
    indent + "# Modal: CUDA Error 803 fix — cuInit via driver API clears sticky runtime error\n"
    + indent + "try:\n"
    + indent + "    import ctypes as _ct\n"
    + indent + '    _ct.CDLL("libcuda.so").cuInit(0)\n'
    + indent + '    _ct.CDLL("libcudart.so").cudaGetLastError()  # clears sticky error state\n'
    + indent + "except Exception:\n"
    + indent + "    pass\n"
)

f.write_text(code[:idx] + insert + code[idx:])
print("patched model_runner.py: inserted cuInit(0) before set_device()")
