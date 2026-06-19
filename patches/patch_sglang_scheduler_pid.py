#!/usr/bin/env python3
"""Patch SGLang scheduler.py to fix LD_LIBRARY_PATH and bypass PyTorch fork CUDA guards.

v5 — LD_LIBRARY_PATH FIX (mirrors sitecustomize v3)
====================================================

Root cause (confirmed 2026-06-19, run bc4gvqrl8):
  The scheduler subprocess runs with:
    LD_LIBRARY_PATH = '/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:...'
  /usr/local/cuda/compat/libcuda.so.1 (compat lib in the CUDA 12.8 image) is loaded FIRST.
  This library's cuInit(0) returns 803 on Modal's cluster (version mismatch with kernel module).
  torch.cuda.is_available() → cudaGetDeviceCount() → internal cuInit(0) via compat → 803 cached.
  Subsequent set_device(0) → 803 from cache → scheduler crashes.

v4 mistake: v4 loaded the compat lib via ctypes and called cuInit → 803 → poisoned linker cache.
  The Python-level torch.cuda._initialized=True bypass didn't help because the C runtime
  libcudart still had 803 cached at the C level; _cuda_setDevice() hit it regardless.

v5 fix (belt-and-suspenders over sitecustomize v3):
  At the TOP of run_scheduler_process(), before any CUDA import:
  1. Reorder LD_LIBRARY_PATH: remove compat, put /usr/local/nvidia/lib64 first.
  2. Load /usr/local/nvidia/lib64/libcuda.so.1 (host driver) via ctypes.
     This registers the host driver as canonical 'libcuda.so.1' in the linker.
  3. Call cuInit(0) on the host driver → 0 = success, 100 = no GPU (build machine).
  4. Keep the PyTorch fork-guard bypasses as an extra safety net.
  NEVER load /usr/local/cuda/compat/libcuda.so.1 — it poisons the linker.
"""

import sys
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/managers/scheduler.py")
GUARD = "modal_fork_bad_fork_v5"
OLD_GUARDS = [
    "modal_fork_bad_fork_v4",
    "modal_fork_bad_fork_v3",
    "modal_fork_bad_fork_v2",
    "modal_fork_bad_fork_v1",
]

f = Path(TARGET_FILE)
if not f.exists():
    print(f"[{GUARD}] {TARGET_FILE} not found, skipping")
    sys.exit(0)

code = f.read_text()

if GUARD in code:
    print(f"[{GUARD}] already patched, skipping")
    sys.exit(0)

# Strip any older version of this patch before inserting v5.
for old in OLD_GUARDS:
    if old in code:
        lines = code.splitlines(keepends=True)
        new_lines, in_block = [], False
        for line in lines:
            if old in line:
                in_block = True
            if not in_block:
                new_lines.append(line)
            # Old v3/v4 block ends at '_initialized = True'; v5 block ends at 'del _os'
            if in_block and ("_initialized = True" in line or "del _os" in line):
                in_block = False  # next line is original scheduler code
        code = "".join(new_lines)
        print(f"[{GUARD}] stripped old patch ({old}) from {TARGET_FILE}")
        break

TARGET_FN = "def run_scheduler_process("
if TARGET_FN not in code:
    print(f"[{GUARD}] function not found in {TARGET_FILE}")
    sys.exit(1)

# Find end of the function signature (closing paren + colon + newline)
idx = code.index(TARGET_FN)
pos = idx + len(TARGET_FN)
depth = 1
while pos < len(code) and depth > 0:
    ch = code[pos]
    if ch == '(':
        depth += 1
    elif ch == ')':
        depth -= 1
    pos += 1
# Advance past ':' to end of line
while pos < len(code) and code[pos] != '\n':
    pos += 1
pos += 1  # skip the newline -- now at first char of function body

# The RESET block: fix LD_LIBRARY_PATH then bypass torch CUDA fork guards.
# Strings with {_var} are f-string LITERALS in the inserted code (evaluated at runtime
# in the scheduler process). This outer string is NOT an f-string; {_var} is literal text.
RESET = (
    f"    # {GUARD}: fix LD_LIBRARY_PATH + bypass PyTorch fork CUDA guards.\n"
    "    import os as _os, ctypes as _ctypes\n"
    "    _nvidia_lib64 = '/usr/local/nvidia/lib64'\n"
    "    _compat_path  = '/usr/local/cuda/compat'\n"
    "    _ld_orig = _os.environ.get('LD_LIBRARY_PATH', '')\n"
    "    _parts = [p for p in _ld_orig.split(':') if p and p != _compat_path]\n"
    "    if _nvidia_lib64 in _parts:\n"
    "        _parts.remove(_nvidia_lib64)\n"
    "    _parts.insert(0, _nvidia_lib64)\n"
    "    _ld_fixed = ':'.join(_parts)\n"
    "    _os.environ['LD_LIBRARY_PATH'] = _ld_fixed\n"
    "    print(f'[sched_v5 pid={_os.getpid()}]"
    " LD_LIBRARY_PATH: {_ld_orig!r} -> {_ld_fixed!r}', flush=True)\n"
    "    print(f'[sched_v5 pid={_os.getpid()}]"
    " CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}', flush=True)\n"
    "    # Load host driver libcuda.so.1 to register it as canonical 'libcuda.so.1'.\n"
    "    # DO NOT load /usr/local/cuda/compat/libcuda.so.1 — that poisons the linker.\n"
    "    _host_libcuda = _nvidia_lib64 + '/libcuda.so.1'\n"
    "    try:\n"
    "        _libcuda = _ctypes.CDLL(_host_libcuda)\n"
    "        _libcuda.cuInit.restype = _ctypes.c_int\n"
    "        _ret = _libcuda.cuInit(0)\n"
    "        print(f'[sched_v5] cuInit(0) via host driver = {_ret}', flush=True)\n"
    "    except OSError as _e:\n"
    "        print(f'[sched_v5] host driver not found: {_e}', flush=True)\n"
    "    import torch.cuda as _torch_cuda\n"
    "    _torch_cuda._is_in_bad_fork = lambda: False\n"
    "    _torch_cuda._initialized = True\n"
    "    del _os, _ctypes, _nvidia_lib64, _compat_path, _ld_orig, _parts, _ld_fixed, _host_libcuda\n"
)

f.write_text(code[:pos] + RESET + code[pos:])
print(f"[{GUARD}] patched {TARGET_FILE}: LD_LIBRARY_PATH fix + fork CUDA guard bypass")
