#!/usr/bin/env python3
"""Inject LD_LIBRARY_PATH fix + host-driver cuInit into sitecustomize.py.

v4 — GLOB-BASED LIBCUDA DISCOVERY (root cause: no libcuda.so.1 symlink on Modal)
==================================================================================

Diagnosis (run ber9wcskc, 2026-06-19):
  /usr/local/nvidia/lib64/libcuda.so.1 does NOT EXIST on Modal GPU workers.
  The NVIDIA container toolkit mounts the versioned driver file (e.g.,
  libcuda.so.560.28.03) but NOT the .so.1 symlink. So v3's explicit path
  ctypes.CDLL('/usr/local/nvidia/lib64/libcuda.so.1') fails with "no such file".
  The dynamic linker then falls back to /etc/ld.so.cache (built at image creation
  time by ldconfig, which scanned /usr/local/cuda/compat). It finds the compat
  libcuda.so.1 there → cuInit=803 → sticky error cached in libcudart → 803 on
  every subsequent CUDA call.

v4 fix:
  1. Fix LD_LIBRARY_PATH (remove compat, put nvidia/lib64 first) — same as v3.
  2. Glob for /usr/local/nvidia/lib64/libcuda.so.* to find the versioned file.
  3. Load it via ctypes.CDLL(versioned_path). glibc reads the ELF SONAME header
     ('libcuda.so.1') and registers this library under that SONAME in the process-
     wide dynamic linker cache. Subsequent dlopen('libcuda.so.1') calls from
     libcudart / torch / sglang return the cached host driver — bypassing both
     ld.so.cache and any embedded RPATH in libcudart.
  4. Set LD_PRELOAD to the versioned path in os.environ. Child processes (the
     SGLang scheduler subprocess) inherit this and glibc's dynamic linker loads
     the host driver FIRST (before Python even starts, before sitecustomize runs
     in the child) — guaranteeing the correct libcuda is canonical.
"""

import sys
from pathlib import Path

GUARD = "cuinit_sitecustomize_v4"

SITECUSTOMIZE_CONTENT = f'''\
# injected by patch_inject_cuinit_sitecustomize.py ({GUARD})
# Fix LD_LIBRARY_PATH + preload host CUDA driver before any Python imports.
# See patch file for full diagnosis.
import os as _os, ctypes as _ctypes, glob as _glob

_nvidia_lib64 = '/usr/local/nvidia/lib64'
_compat_path  = '/usr/local/cuda/compat'

# Step 1: Remove compat from LD_LIBRARY_PATH, put host driver dir first.
_ld_orig = _os.environ.get('LD_LIBRARY_PATH', '')
_parts = [p for p in _ld_orig.split(':') if p and p != _compat_path]
if _nvidia_lib64 in _parts:
    _parts.remove(_nvidia_lib64)
_parts.insert(0, _nvidia_lib64)
_ld_fixed = ':'.join(_parts)
_os.environ['LD_LIBRARY_PATH'] = _ld_fixed
print(f'[cuinit_site_v4 pid={{_os.getpid()}}] LD_LIBRARY_PATH: {{_ld_orig!r}} -> {{_ld_fixed!r}}', flush=True)

# Step 2: Find versioned libcuda.so.* — Modal does NOT create libcuda.so.1 symlink,
# only the versioned file (e.g. libcuda.so.560.28.03).
_candidates = sorted(_glob.glob(_nvidia_lib64 + '/libcuda.so*'))
_real = [f for f in _candidates if _os.path.isfile(f)]
_host_libcuda = _real[0] if _real else None

if _host_libcuda:
    try:
        # Step 3: Load versioned file. glibc reads its ELF SONAME ('libcuda.so.1')
        # and registers it in the process-wide SONAME cache. All subsequent
        # dlopen('libcuda.so.1') calls get the host driver — ignoring ld.so.cache.
        _lib = _ctypes.CDLL(_host_libcuda)
        _lib.cuInit.restype = _ctypes.c_int
        _ret = _lib.cuInit(0)
        # 0 = success, 100 = CUDA_ERROR_NO_DEVICE (build machine), else = problem
        print(f'[cuinit_site_v4 pid={{_os.getpid()}}] cuInit(0) via {{_host_libcuda}} = {{_ret}}', flush=True)
        # Step 4: Propagate to child processes via LD_PRELOAD (scheduler subprocess).
        # In children, glibc processes LD_PRELOAD before Python starts, ensuring the
        # host driver is first in the SONAME cache before any imports.
        _preload = _os.environ.get('LD_PRELOAD', '')
        if _host_libcuda not in _preload:
            _os.environ['LD_PRELOAD'] = (_host_libcuda + (':' + _preload if _preload else ''))
            print(f'[cuinit_site_v4 pid={{_os.getpid()}}] LD_PRELOAD -> {{_os.environ["LD_PRELOAD"]!r}}', flush=True)
    except OSError as _e:
        print(f'[cuinit_site_v4 pid={{_os.getpid()}}] CDLL({{_host_libcuda!r}}) failed: {{_e}}', flush=True)
else:
    _all_files = _os.listdir(_nvidia_lib64) if _os.path.isdir(_nvidia_lib64) else []
    print(f'[cuinit_site_v4 pid={{_os.getpid()}}] no libcuda.so* in {{_nvidia_lib64!r}}; listing: {{_all_files[:30]}}', flush=True)

del _nvidia_lib64, _compat_path, _ld_orig, _parts, _ld_fixed, _candidates, _real, _host_libcuda, _os, _ctypes, _glob
'''

# Find the site-packages directory for the current Python
import site
candidates = site.getsitepackages() if hasattr(site, "getsitepackages") else []
candidates.append(
    f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
)

target = None
for sp in candidates:
    p = Path(sp)
    if p.is_dir():
        target = p / "sitecustomize.py"
        break

if target is None:
    print("Could not find site-packages, skipping")
    sys.exit(0)

if target.exists():
    existing = target.read_text()
    if GUARD in existing:
        print(f"sitecustomize.py already has {GUARD} injection at {target}")
        sys.exit(0)
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"replaced sitecustomize.py with {GUARD} at {target} (was: {len(existing)} chars)")
else:
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"wrote {GUARD} to {target}")
