#!/usr/bin/env python3
"""Patch SGLang scheduler.py to fix LD_LIBRARY_PATH and bypass PyTorch fork CUDA guards.

v6 — GLOB-BASED LIBCUDA DISCOVERY (mirrors sitecustomize v4)
=============================================================

Root cause (confirmed 2026-06-19, run ber9wcskc):
  /usr/local/nvidia/lib64/libcuda.so.1 does NOT EXIST on Modal GPU workers.
  Modal's NVIDIA container toolkit mounts only the versioned file (e.g.,
  libcuda.so.560.28.03). v5 tried to load libcuda.so.1 directly → failed →
  glibc fell back to /etc/ld.so.cache which had the compat library cached →
  cuInit=803 → libcudart sticky error → set_device crashes.

v6 fix (belt-and-suspenders over sitecustomize v4 + LD_PRELOAD):
  sitecustomize v4 runs first (at Python startup in the scheduler subprocess),
  loading the versioned libcuda and setting LD_PRELOAD. This should fix 803
  before run_scheduler_process() is even called. This patch is kept as a safety
  net in case sitecustomize didn't run (e.g., pre-forked Ray worker pool).

  At the TOP of run_scheduler_process():
  1. Fix LD_LIBRARY_PATH (remove compat, put nvidia/lib64 first).
  2. Glob for /usr/local/nvidia/lib64/libcuda.so.* to find the versioned file.
  3. Load it via ctypes.CDLL — registers it as canonical 'libcuda.so.1' in the
     process-wide SONAME cache.
  4. Set LD_PRELOAD for further child processes.
  5. Bypass PyTorch fork-CUDA guards as belt-and-suspenders.
"""

import sys
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/managers/scheduler.py")
GUARD = "modal_fork_bad_fork_v6"
OLD_GUARDS = [
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

# The RESET block: glob-based libcuda discovery + LD_LIBRARY_PATH fix + fork guards.
# Outer string is NOT an f-string; {_var} are literal brace pairs for the inserted code.
RESET = (
    f"    # {GUARD}: fix LD_LIBRARY_PATH + glob-based host driver preload.\n"
    "    import os as _os, ctypes as _ctypes, glob as _glob\n"
    "    _nvidia_lib64 = '/usr/local/nvidia/lib64'\n"
    "    _compat_path  = '/usr/local/cuda/compat'\n"
    "    _ld_orig = _os.environ.get('LD_LIBRARY_PATH', '')\n"
    "    _parts = [p for p in _ld_orig.split(':') if p and p != _compat_path]\n"
    "    if _nvidia_lib64 in _parts:\n"
    "        _parts.remove(_nvidia_lib64)\n"
    "    _parts.insert(0, _nvidia_lib64)\n"
    "    _ld_fixed = ':'.join(_parts)\n"
    "    _os.environ['LD_LIBRARY_PATH'] = _ld_fixed\n"
    "    print(f'[sched_v6 pid={_os.getpid()}]"
    " LD_LIBRARY_PATH: {_ld_orig!r} -> {_ld_fixed!r}', flush=True)\n"
    "    print(f'[sched_v6 pid={_os.getpid()}]"
    " CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}', flush=True)\n"
    "    # Glob for versioned libcuda.so.* (libcuda.so.1 symlink may not exist).\n"
    "    _candidates = sorted(_glob.glob(_nvidia_lib64 + '/libcuda.so*'))\n"
    "    _real = [f for f in _candidates if _os.path.isfile(f)]\n"
    "    _host_libcuda = _real[0] if _real else None\n"
    "    if _host_libcuda:\n"
    "        try:\n"
    "            _lib = _ctypes.CDLL(_host_libcuda)\n"
    "            _lib.cuInit.restype = _ctypes.c_int\n"
    "            _ret = _lib.cuInit(0)\n"
    "            print(f'[sched_v6] cuInit(0) via {_host_libcuda} = {_ret}', flush=True)\n"
    "            _preload = _os.environ.get('LD_PRELOAD', '')\n"
    "            if _host_libcuda not in _preload:\n"
    "                _os.environ['LD_PRELOAD'] = (_host_libcuda +"
    " (':' + _preload if _preload else ''))\n"
    "        except OSError as _e:\n"
    "            print(f'[sched_v6] CDLL({_host_libcuda!r}) failed: {_e}', flush=True)\n"
    "    else:\n"
    "        _all_files = _os.listdir(_nvidia_lib64) if _os.path.isdir(_nvidia_lib64) else []\n"
    "        print(f'[sched_v6] no libcuda.so* in {_nvidia_lib64!r}; listing: {_all_files[:30]}', flush=True)\n"
    "    import torch.cuda as _torch_cuda\n"
    "    _torch_cuda._is_in_bad_fork = lambda: False\n"
    "    _torch_cuda._initialized = True\n"
    "    del _os, _ctypes, _glob, _nvidia_lib64, _compat_path, _ld_orig, _parts, _ld_fixed,"
    " _candidates, _real, _host_libcuda\n"
)

f.write_text(code[:pos] + RESET + code[pos:])
print(f"[{GUARD}] patched {TARGET_FILE}: glob-based libcuda preload + fork CUDA guard bypass")
