#!/usr/bin/env python3
"""Patch SGLang engine.py to use fork instead of spawn for scheduler subprocess.

Context: On Modal, GPU device access is a per-process OS privilege assigned by
the Raylet to Ray actor processes. mp.Process(spawn) creates a fresh interpreter
without GPU access (cuInit → 803). mp.Process(fork) inherits it.

This patches _set_envs_and_config() in engine.py which calls
  mp.set_start_method("spawn", force=True)
just before the scheduler subprocess is started. This runs inside the HTTP server
process (which is itself a forked child of the SGLangEngine Ray actor, via the
companion patch patch_launch_server_fork.py). If we don't also patch here, the
scheduler would still be spawned with start_method="spawn".

After both patches:
  Ray actor --fork--> HTTP server --fork--> scheduler
Each inherits the previous process's GPU device access.
"""
import sys
import re
from pathlib import Path

TARGET_FILE = (sys.argv[1] if len(sys.argv) > 1
               else "/root/sglang/python/sglang/srt/entrypoints/engine.py")
GUARD = "modal_fork_scheduler_v1"

f = Path(TARGET_FILE)
if not f.exists():
    print(f"[modal_fork_scheduler_v1] {TARGET_FILE} not found, skipping")
    sys.exit(0)

code = f.read_text()

if GUARD in code:
    print(f"[modal_fork_scheduler_v1] already patched, skipping")
    sys.exit(0)

# Match the exact call: mp.set_start_method("spawn", force=True)
# This appears inside _set_envs_and_config().
OLD = 'mp.set_start_method("spawn", force=True)'
NEW = f'mp.set_start_method("fork", force=True)  # {GUARD}: inherit Ray actor GPU cgroup/fds'

if OLD not in code:
    print(f"[modal_fork_scheduler_v1] target string not found: {OLD!r}")
    # Print all set_start_method calls for diagnosis
    for i, line in enumerate(code.splitlines(), 1):
        if "set_start_method" in line:
            print(f"  line {i}: {line!r}")
    sys.exit(1)

count = code.count(OLD)
code = code.replace(OLD, NEW)  # replace all occurrences
f.write_text(code)
print(f"[modal_fork_scheduler_v1] patched {TARGET_FILE} ({count} occurrence(s)): spawn → fork")
