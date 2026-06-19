#!/usr/bin/env python3
"""Patch SGLang scheduler.py to fix LD_LIBRARY_PATH and bypass PyTorch fork CUDA guards.

v8 — ADD cudaGetLastError() TO CLEAR PARENT'S STICKY 803
=========================================================

Key finding from smoke #13 (2026-06-19):
  Host libcuda is at /usr/lib/x86_64-linux-gnu/libcuda.so.580.95.05
  (NOT at /usr/local/nvidia/lib64 — that directory is empty on Modal GPU workers).

  sched_v7 successfully:
    [sched_v7] found host libcuda in '/usr/lib/x86_64-linux-gnu'
    [sched_v7] cuInit(0) via .../libcuda.so.580.95.05 = 0  ← SUCCESS
    [sched_v7] bypassed torch.cuda fork guards

  But model_runner still crashed:
    [modal_v5] host driver not found: /usr/local/nvidia/lib64/libcuda.so.1 ← wrong path
    Traceback: RuntimeError from set_device (sticky 803 still in libcudart)

v8 additions:
  After successful cuInit(0)=0, call cudaGetLastError() on libcudart to clear
  the sticky 803 error that was inherited from the SGLangEngine parent actor
  (which got 803 when it imported torch.cuda without a valid driver context).
  This clears the per-thread error variable in libcudart BEFORE the scheduler
  spawns model_runner subprocesses, so those forks inherit a clean state.
"""

import sys
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/managers/scheduler.py")
GUARD = "modal_fork_bad_fork_v8"
OLD_GUARDS = [
    "modal_fork_bad_fork_v7",
    "modal_fork_bad_fork_v6",
    "modal_fork_bad_fork_v5",
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

# Strip any older version of this patch before inserting v6.
for old in OLD_GUARDS:
    if old in code:
        lines = code.splitlines(keepends=True)
        new_lines, in_block = [], False
        for line in lines:
            if old in line:
                in_block = True
            if not in_block:
                new_lines.append(line)
            # Old blocks end at 'del _os' or '_initialized = True'
            if in_block and ("del _os" in line or "_initialized = True" in line):
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
pos += 1  # skip the newline — now at first char of function body

# The RESET block: broad libcuda search + LD_LIBRARY_PATH fix + clear sticky 803 + fork guards.
# v8: After cuInit(0)=0, call cudaGetLastError() on libcudart to clear the sticky 803
# that was inherited from the SGLangEngine actor (which got 803 when importing torch.cuda).
# This is needed so that model_runner subprocesses (forked from the scheduler) inherit
# a clean libcudart state, allowing torch.cuda.set_device() to succeed.
RESET = (
    f"    # {GUARD}: fix LD_LIBRARY_PATH + broad host driver search + clear sticky 803.\n"
    "    import os as _os, ctypes as _ctypes, glob as _glob, sys as _sys\n"
    "    _compat_path = '/usr/local/cuda/compat'\n"
    "    _ld_orig = _os.environ.get('LD_LIBRARY_PATH', '')\n"
    "    _parts = [p for p in _ld_orig.split(':') if p and p != _compat_path]\n"
    "    _ld_fixed = ':'.join(_parts)\n"
    "    _os.environ['LD_LIBRARY_PATH'] = _ld_fixed\n"
    "    print(f'[sched_v8 pid={_os.getpid()}]"
    " LD_LIBRARY_PATH: {_ld_orig!r} -> {_ld_fixed!r}', flush=True)\n"
    "    print(f'[sched_v8 pid={_os.getpid()}]"
    " CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}', flush=True)\n"
    "    # Broad search: /usr/lib/x86_64-linux-gnu confirmed from smoke13.\n"
    "    _SEARCH_PATHS = [\n"
    "        '/usr/lib/x86_64-linux-gnu', '/usr/local/nvidia/lib64', '/usr/local/nvidia/lib',\n"
    "        '/usr/lib64', '/usr/lib', '/usr/local/lib',\n"
    "    ] + [p for p in _parts if p]\n"
    "    _host_libcuda = None\n"
    "    for _sdir in _SEARCH_PATHS:\n"
    "        _cands = sorted(_glob.glob(_sdir + '/libcuda.so*'))\n"
    "        _real = [f for f in _cands if _os.path.isfile(f) and _compat_path not in f]\n"
    "        if _real:\n"
    "            _host_libcuda = _real[-1]\n"
    "            print(f'[sched_v8] found host libcuda in {_sdir!r}: {_real}', flush=True)\n"
    "            break\n"
    "    if not _host_libcuda:\n"
    "        print(f'[sched_v8] no host libcuda found in any search path', flush=True)\n"
    "    if _host_libcuda:\n"
    "        try:\n"
    "            _lib = _ctypes.CDLL(_host_libcuda)\n"
    "            _lib.cuInit.restype = _ctypes.c_int\n"
    "            _ret = _lib.cuInit(0)\n"
    "            print(f'[sched_v8] cuInit(0) via {_host_libcuda} = {_ret}', flush=True)\n"
    "            _preload = _os.environ.get('LD_PRELOAD', '')\n"
    "            if _host_libcuda not in _preload:\n"
    "                _os.environ['LD_PRELOAD'] = (_host_libcuda +"
    " (':' + _preload if _preload else ''))\n"
    "        except OSError as _e:\n"
    "            print(f'[sched_v8] CDLL({_host_libcuda!r}) failed: {_e}', flush=True)\n"
    "    # Clear sticky 803 inherited from SGLangEngine actor (got 803 at torch.cuda import).\n"
    "    # cudaGetLastError() resets the per-thread error variable in libcudart.\n"
    "    try:\n"
    "        _cudart = _ctypes.CDLL('libcudart.so.12')\n"
    "        _cudart.cudaGetLastError.restype = _ctypes.c_int\n"
    "        _sticky = _cudart.cudaGetLastError()\n"
    "        print(f'[sched_v8] cudaGetLastError() cleared sticky error: {_sticky}', flush=True)\n"
    "    except Exception as _cue:\n"
    "        print(f'[sched_v8] cudaGetLastError failed: {_cue}', flush=True)\n"
    "    # Only bypass torch.cuda fork-guard if torch.cuda is ALREADY in sys.modules.\n"
    "    # Do NOT import it here — that triggers the sticky 803 before libcuda is fixed.\n"
    "    _tc = _sys.modules.get('torch.cuda')\n"
    "    if _tc is not None:\n"
    "        _tc._is_in_bad_fork = lambda: False\n"
    "        _tc._initialized = True\n"
    "        print(f'[sched_v8] bypassed torch.cuda fork guards', flush=True)\n"
    "    else:\n"
    "        print(f'[sched_v8] torch.cuda not yet imported; fork guards skipped', flush=True)\n"
    "    del _os, _ctypes, _glob, _sys, _compat_path, _ld_orig, _parts, _ld_fixed,"
    " _SEARCH_PATHS, _host_libcuda, _tc\n"
)

f.write_text(code[:pos] + RESET + code[pos:])
print(f"[{GUARD}] patched {TARGET_FILE}: broad libcuda search + cudaGetLastError + no torch.cuda import")
