#!/usr/bin/env python3
"""Patch SGLang scheduler.py to (a) call cuInit via compat library and (b) bypass
PyTorch's fork CUDA guards in run_scheduler_process.

v4: add cuInit(0) via the NVIDIA forward-compat libcuda.so.1 and cudaGetLastError()
to clear any inherited sticky-803 state from the parent process (HTTP server).

Root cause chain:
  1. SGLangEngine Ray actor (spawned by Ray) imports sglang at startup.
     If libcuda.so.1 at CUDA runtime init time is the host version (too old for
     CUDA 12.8), the first CUDA call returns 803, poisoning libcudart's internal
     "init failed" cache.
  2. HTTP server is FORKED from the SGLangEngine actor -> inherits the poisoned state.
  3. Scheduler is FORKED from HTTP server -> inherits the poisoned state.
  4. cudaSetDevice(0) in init_torch_distributed -> 803 (cached init failure).

v4 two-layer fix:
  A. patch_inject_cuinit_sitecustomize.py (companion patch, applied earlier) runs
     cuInit(0) via the compat library at interpreter startup in the SGLangEngine Ray
     actor. This fixes the root cause: actor CUDA state is "init ok", and the fork
     chain propagates that clean state to HTTP server and scheduler.
  B. This patch adds belt-and-suspenders in run_scheduler_process itself:
     - cuInit(0) via compat libcuda.so.1 (absolute path, bypasses LD_LIBRARY_PATH)
     - cudaGetLastError() to clear any per-thread sticky 803 error
     - Diagnostic prints: LD_LIBRARY_PATH, CUDA_VISIBLE_DEVICES, cuInit result,
       cudaGetLastError result - visible in Modal task log

PyTorch fork CUDA guard bypass (unchanged from v3):
  Layer 1: torch.cuda._is_in_bad_fork = lambda: False
    -- stops the Python-level check in _lazy_init().
  Layer 2: torch.cuda._initialized = True
    -- makes _lazy_init() return early without calling torch._C._cuda_init(),
       which has C++ TORCH_INTERNAL_ASSERT(!is_device_in_bad_fork) at Module.cpp:1498.
  After these bypasses, CUDA is accessed at C++ ATen level by cudaSetDevice() directly.
"""

import sys
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/managers/scheduler.py")
GUARD = "modal_fork_bad_fork_v4"
OLD_GUARDS = ["modal_fork_bad_fork_v3", "modal_fork_bad_fork_v2", "modal_fork_bad_fork_v1"]

f = Path(TARGET_FILE)
if not f.exists():
    print(f"[{GUARD}] {TARGET_FILE} not found, skipping")
    sys.exit(0)

code = f.read_text()

if GUARD in code:
    print(f"[{GUARD}] already patched, skipping")
    sys.exit(0)

# Strip any older version of this patch before inserting v4.
for old in OLD_GUARDS:
    if old in code:
        # The old RESET block was inserted just after the function signature line.
        # Find and remove lines from the old guard comment up to (not including)
        # the first non-blank, non-comment line that was already in the file.
        # Simple approach: remove lines containing the old guard markers.
        lines = code.splitlines(keepends=True)
        new_lines, in_block = [], False
        for line in lines:
            if old in line:
                in_block = True
            if not in_block:
                new_lines.append(line)
            # End of inserted block: first line that doesn't start with spaces+# or spaces
            # and doesn't have our guard markers. Use the torch.cuda._initialized line as sentinel.
            if in_block and "_initialized = True" in line:
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

# NOTE: strings inside RESET that begin with f' are f-string LITERALS in the
# inserted code (evaluated at runtime in the scheduler process). The outer Python
# strings here are regular strings (not f-prefixed), so {_libname!r} etc. are
# just literal text that forms the inserted f-string. Only the first line uses
# an outer f"..." to substitute GUARD.
RESET = (
    f"    # {GUARD}: cuInit via compat + clear sticky error + bypass fork guards.\n"
    "    import os as _os, ctypes as _ctypes\n"
    "    print(f'[sched_cuda_v4 pid={_os.getpid()}]"
    " LD_LIBRARY_PATH={_os.environ.get(\"LD_LIBRARY_PATH\")!r}', flush=True)\n"
    "    print(f'[sched_cuda_v4 pid={_os.getpid()}]"
    " CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}', flush=True)\n"
    "    _cu_init_ok = False\n"
    "    for _libname in (\n"
    "        '/usr/local/cuda/compat/libcuda.so.1',\n"
    "        '/usr/local/cuda-12.8/compat/libcuda.so.1',\n"
    "        'libcuda.so.1', 'libcuda.so',\n"
    "    ):\n"
    "        try:\n"
    "            _libcuda = _ctypes.CDLL(_libname)\n"
    "            _libcuda.cuInit.restype = _ctypes.c_int\n"
    "            _ret = _libcuda.cuInit(0)\n"
    "            print(f'[sched_cuda_v4] cuInit via {_libname!r} = {_ret}', flush=True)\n"
    "            if _ret == 0:\n"
    "                _cu_init_ok = True\n"
    "                break\n"
    "        except OSError as _e:\n"
    "            print(f'[sched_cuda_v4] {_libname!r}: {_e}', flush=True)\n"
    "    for _rtlib in ('libcudart.so.12', 'libcudart.so'):\n"
    "        try:\n"
    "            _rt = _ctypes.CDLL(_rtlib)\n"
    "            _rt.cudaGetLastError.restype = _ctypes.c_int\n"
    "            _rt_err = _rt.cudaGetLastError()\n"
    "            print(f'[sched_cuda_v4] cudaGetLastError via {_rtlib} = {_rt_err}', flush=True)\n"
    "            break\n"
    "        except OSError:\n"
    "            continue\n"
    "    import torch.cuda as _torch_cuda\n"
    "    _torch_cuda._is_in_bad_fork = lambda: False\n"
    "    _torch_cuda._initialized = True\n"
)

f.write_text(code[:pos] + RESET + code[pos:])
print(f"[{GUARD}] patched {TARGET_FILE}: cuInit + cudaGetLastError + fork CUDA guard bypass")
