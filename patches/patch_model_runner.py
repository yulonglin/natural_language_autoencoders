#!/usr/bin/env python3
"""Patch SGLang model_runner.py: diagnostic print at init_torch_distributed.

v5 — DIAGNOSTIC ONLY (LD_LIBRARY_PATH fix moved upstream to sitecustomize + scheduler)
========================================================================================

With sitecustomize v3 + scheduler v5 fixing LD_LIBRARY_PATH upstream, by the time we
reach init_torch_distributed(), the host driver's libcuda.so.1 should already be loaded
and cuInit = 0. This patch only prints confirmation.

If something still fails here (set_device raises 803), the prints tell us exactly what
LD_LIBRARY_PATH and CUDA_VISIBLE_DEVICES the scheduler process has at that point.

NEVER load /usr/local/cuda/compat/libcuda.so.1 here — that would poison the linker.
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
GUARD = "modal_cuda_803_fix_v5"

if GUARD in code:
    print("model_runner.py already patched (v5)")
    sys.exit(0)

# Remove any older version of this patch before inserting v5.
for old_guard in (
    "modal_cuda_803_fix_v4",
    "modal_cuda_803_fix_v3",
    "modal_cuda_803_fix_v2",
    "modal_cuda_803_fix_v1",
):
    if old_guard in code:
        lines = code.splitlines(keepends=True)
        new_lines = []
        in_block = False
        for line in lines:
            if old_guard in line:
                in_block = True
            if not in_block:
                new_lines.append(line)
            if in_block and TARGET in line:
                new_lines.append(line)
                in_block = False
        code = "".join(new_lines)
        print(f"stripped old patch ({old_guard}) from model_runner.py")
        break

if TARGET not in code:
    print("target not found in model_runner.py")
    sys.exit(1)

idx = code.index(TARGET)
line_start = code.rfind("\n", 0, idx) + 1
indent = code[line_start:idx]

insert = (
    indent + f"# {GUARD}: diagnostic print at init_torch_distributed.\n"
    + indent + "import os as _os, ctypes as _ctypes\n"
    + indent + "print(f'[modal_v5 pid={_os.getpid()}]"
    " LD_LIBRARY_PATH={_os.environ.get(\"LD_LIBRARY_PATH\")!r}', flush=True)\n"
    + indent + "print(f'[modal_v5 pid={_os.getpid()}]"
    " CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}"
    " gpu_id={self.gpu_id}', flush=True)\n"
    # Call cuInit on host driver to confirm it's registered and returns 0.
    + indent + "_host_libcuda = '/usr/local/nvidia/lib64/libcuda.so.1'\n"
    + indent + "try:\n"
    + indent + "    _libcuda = _ctypes.CDLL(_host_libcuda)\n"
    + indent + "    _libcuda.cuInit.restype = _ctypes.c_int\n"
    + indent + "    _cu_ret = _libcuda.cuInit(0)\n"
    + indent + "    print(f'[modal_v5] cuInit(0) via host driver = {_cu_ret}', flush=True)\n"
    + indent + "except OSError as _e:\n"
    + indent + "    print(f'[modal_v5] host driver not found: {_e}', flush=True)\n"
    + indent + "del _os, _ctypes, _host_libcuda\n"
)

f.write_text(code[:line_start] + insert + code[line_start:])
print("patched model_runner.py (v5): diagnostic print at init_torch_distributed")
