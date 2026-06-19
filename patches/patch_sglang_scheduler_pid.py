#!/usr/bin/env python3
"""Patch SGLang scheduler.py to bypass PyTorch's C-level fork CUDA guard.

Context: After the fork-chain fix (patch_launch_server_fork + patch_sglang_mp_fork),
the scheduler subprocess is a FORKED child of the HTTP server. The cgroup GPU device
access IS inherited via fork — but PyTorch's _lazy_init() blocks CUDA init in forked
subprocesses via a C-level check:

    # torch/cuda/__init__.py line ~52:
    _is_in_bad_fork = getattr(torch._C, "_cuda_isInBadFork", lambda: False)

    # in _lazy_init():
    if _is_in_bad_fork():
        raise RuntimeError("Cannot re-initialize CUDA in forked subprocess...")

Unlike the old _original_pid Python check, _cuda_isInBadFork is a C builtin that
cannot be bypassed by resetting Python-level variables.

Fix (two layers):
  1. torch.cuda._is_in_bad_fork = lambda: False  — bypasses the Python guard in _lazy_init.
  2. torch.cuda._initialized = True              — makes _lazy_init() return early so
     torch._C._cuda_init() (which also has a C++ TORCH_INTERNAL_ASSERT(!is_device_in_bad_fork)
     inside Module.cpp) is NEVER called. CUDA is then lazily initialized at the C++ ATen
     level by the first cudaSetDevice() call, which has NO bad-fork check and succeeds
     because GPU access is inherited via Modal's fork() (cgroup/FD membership is preserved).
"""

import sys
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/managers/scheduler.py")
GUARD = "modal_fork_bad_fork_v3"

f = Path(TARGET_FILE)
if not f.exists():
    print(f"[{GUARD}] {TARGET_FILE} not found, skipping")
    sys.exit(0)

code = f.read_text()

if GUARD in code:
    print(f"[{GUARD}] already patched, skipping")
    sys.exit(0)

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
# Advance to end of line (past the ':')
while pos < len(code) and code[pos] != '\n':
    pos += 1
pos += 1  # skip the newline — now at first char of function body

RESET = (
    f"    # {GUARD}: bypass ALL of PyTorch's fork CUDA guards.\n"
    "    # Layer 1: _is_in_bad_fork lambda stops the Python-level check in _lazy_init.\n"
    "    # Layer 2: _initialized=True makes _lazy_init() return early, never calling\n"
    "    #   torch._C._cuda_init() which has a C++ TORCH_INTERNAL_ASSERT(!is_device_in_bad_fork).\n"
    "    # CUDA is then lazily initialized at the C++ ATen level by the first cudaSetDevice()\n"
    "    # call (no bad-fork check there), which succeeds because GPU access is inherited\n"
    "    # via fork() in Modal's container (cgroup/FD membership is preserved by fork).\n"
    "    import torch.cuda as _torch_cuda\n"
    "    _torch_cuda._is_in_bad_fork = lambda: False\n"
    "    _torch_cuda._initialized = True\n"
)

f.write_text(code[:pos] + RESET + code[pos:])
print(f"[{GUARD}] patched {TARGET_FILE}: bypassed all fork CUDA guards in run_scheduler_process")
