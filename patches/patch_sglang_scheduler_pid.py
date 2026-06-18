#!/usr/bin/env python3
"""Patch SGLang scheduler.py to reset torch.cuda._original_pid at process start.

Context: After the fork-chain fix (patch_launch_server_fork + patch_sglang_mp_fork),
the scheduler subprocess is a FORKED child of the HTTP server, which is itself a
forked child of the SGLangEngine Ray actor. The cgroup GPU device access IS inherited
via fork — but PyTorch's _lazy_init() has a software guard:

    if _original_pid is None:
        _original_pid = os.getpid()      # first caller: sets it
    elif _original_pid != os.getpid():   # forked child: mismatch → RuntimeError
        raise RuntimeError(
            "Cannot re-initialize CUDA in forked subprocess. ..."
        )

The scheduler's _original_pid was set in the SGLangEngine Ray actor (the original
process that imported torch.cuda) and inherited via fork. The scheduler's os.getpid()
differs → RuntimeError, even though the scheduler genuinely HAS GPU access.

Fix: reset _original_pid = os.getpid() at the very top of run_scheduler_process(),
before any CUDA call. _lazy_init() then sees a match and proceeds to torch._C._cuda_init(),
which succeeds because the cgroup membership was inherited via the fork chain.
"""

import sys
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/managers/scheduler.py")
GUARD = "modal_fork_pid_reset_v1"

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
    f"    # {GUARD}: bypass PyTorch's fork-CUDA guard; GPU access is inherited via cgroup\n"
    "    import os as _os, torch.cuda as _torch_cuda\n"
    "    _torch_cuda._original_pid = _os.getpid()\n"
)

f.write_text(code[:pos] + RESET + code[pos:])
print(f"[{GUARD}] patched {TARGET_FILE}: inserted _original_pid reset in run_scheduler_process")
