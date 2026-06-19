#!/usr/bin/env python3
"""Patch SGLang model_runner.py: broad libcuda search + clear sticky 803.

v6 — BROAD SEARCH + cudaGetLastError() to clear inherited sticky 803
======================================================================

Root cause chain (confirmed from smoke #13, 2026-06-19):
  1. SGLangEngine Ray actor imports torch.cuda → compat libcuda loads → cuInit=803
     (sticky error set in libcudart's per-thread error variable)
  2. Fork → HTTP server → fork → scheduler subprocess
  3. scheduler (sched_v7/v8) finds real libcuda in /usr/lib/x86_64-linux-gnu/,
     cuInit(0)=0 via driver API, bypasses torch.cuda fork guards, calls
     cudaGetLastError() to clear the inherited sticky 803
  4. Fork → model_runner subprocess (inherits scheduler's cleared state)
  5. model_runner.py v5 tried /usr/local/nvidia/lib64/libcuda.so.1 → NOT FOUND
     → OSError printed, then set_device hit residual 803 → RuntimeError → crash

v6 fix:
  - Search /usr/lib/x86_64-linux-gnu/ FIRST (that's where libcuda.so.580.95.05 is)
  - Load it via ctypes (confirms real driver in SONAME cache for libcudart)
  - Call cudaGetLastError() on libcudart to clear any residual 803 from parent forks
  - THEN allow set_device(gpu_id) to proceed
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
GUARD = "modal_cuda_803_fix_v6"

if GUARD in code:
    print("model_runner.py already patched (v6)")
    sys.exit(0)

# Strip older versions.
for old_guard in (
    "modal_cuda_803_fix_v5",
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
    indent + f"# {GUARD}: broad libcuda search + clear sticky 803 before set_device\n"
    + indent + "import os as _mr_os, ctypes as _mr_ct, glob as _mr_gl\n"
    + indent + "print(f'[modal_v6 pid={_mr_os.getpid()}]"
    " LD={_mr_os.environ.get(\"LD_LIBRARY_PATH\")!r}"
    " CVD={_mr_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}"
    " gpu_id={self.gpu_id}', flush=True)\n"
    # Broad search — /usr/lib/x86_64-linux-gnu/ confirmed from smoke13
    + indent + "_mr_COMPAT = '/usr/local/cuda/compat'\n"
    + indent + "_mr_PATHS = [\n"
    + indent + "    '/usr/lib/x86_64-linux-gnu', '/usr/local/nvidia/lib64',\n"
    + indent + "    '/usr/local/nvidia/lib', '/usr/lib64', '/usr/lib', '/usr/local/lib',\n"
    + indent + "] + [p for p in _mr_os.environ.get('LD_LIBRARY_PATH', '').split(':')\n"
    + indent + "     if p and p != _mr_COMPAT]\n"
    + indent + "_mr_libcuda = None\n"
    + indent + "for _mr_sp in _mr_PATHS:\n"
    + indent + "    _mr_cands = [\n"
    + indent + "        fn for fn in sorted(_mr_gl.glob(_mr_sp + '/libcuda.so*'))\n"
    + indent + "        if _mr_os.path.isfile(fn) and _mr_COMPAT not in fn\n"
    + indent + "    ]\n"
    + indent + "    if _mr_cands:\n"
    + indent + "        _mr_libcuda = _mr_cands[-1]\n"
    + indent + "        break\n"
    + indent + "if _mr_libcuda:\n"
    + indent + "    try:\n"
    + indent + "        _mr_drv = _mr_ct.CDLL(_mr_libcuda)\n"
    + indent + "        _mr_drv.cuInit.restype = _mr_ct.c_int\n"
    + indent + "        _mr_ret = _mr_drv.cuInit(0)\n"
    + indent + "        print(f'[modal_v6] cuInit(0) via {_mr_libcuda} = {_mr_ret}', flush=True)\n"
    + indent + "    except OSError as _mr_e:\n"
    + indent + "        print(f'[modal_v6] CDLL failed: {_mr_e}', flush=True)\n"
    + indent + "else:\n"
    + indent + "    print('[modal_v6] no host libcuda found in any path', flush=True)\n"
    # Clear sticky 803 from parent (SGLangEngine actor inherited by scheduler, then model_runner)
    + indent + "try:\n"
    + indent + "    _mr_rt = _mr_ct.CDLL('libcudart.so.12')\n"
    + indent + "    _mr_rt.cudaGetLastError.restype = _mr_ct.c_int\n"
    + indent + "    _mr_sticky = _mr_rt.cudaGetLastError()\n"
    + indent + "    print(f'[modal_v6] cudaGetLastError() cleared: {_mr_sticky}', flush=True)\n"
    + indent + "except Exception as _mr_ce:\n"
    + indent + "    print(f'[modal_v6] cudaGetLastError failed: {_mr_ce}', flush=True)\n"
    + indent + "print(f'[modal_v6] calling set_device({self.gpu_id})', flush=True)\n"
    + indent + "del _mr_os, _mr_ct, _mr_gl, _mr_COMPAT, _mr_PATHS, _mr_libcuda\n"
)

f.write_text(code[:line_start] + insert + code[line_start:])
print("patched model_runner.py (v6): broad libcuda search + cudaGetLastError before set_device")
