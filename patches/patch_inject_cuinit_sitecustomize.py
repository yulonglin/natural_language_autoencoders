#!/usr/bin/env python3
"""Inject LD_LIBRARY_PATH fix + host-driver cuInit into sitecustomize.py.

v5 — BROAD SEARCH (root cause: libcuda not in /usr/local/nvidia/lib64 on Modal)
================================================================================

v4 diagnosis (smoke #12, 2026-06-19):
  /usr/local/nvidia/lib64 is EMPTY at runtime on Modal GPU workers — glob returns [].
  The compat library at /usr/local/cuda/compat/libcuda.so.1 then loads (from
  ld.so.cache built at image creation time), causing cuInit=803. FSDP actors work
  despite this (they may use NCCL driver API path that avoids 803), but sglang
  scheduler uses torch.cuda runtime API which hits the sticky 803.

v5 fix:
  Search for libcuda.so.* in a MUCH WIDER set of paths:
  - /usr/local/nvidia/lib64 (standard toolkit path — was empty in v4)
  - /usr/local/nvidia/lib   (non-64 variant)
  - /usr/lib/x86_64-linux-gnu/ (system lib path on Debian/Ubuntu)
  - /usr/lib64/              (system lib path on RHEL/CentOS/SUSE)
  - /usr/lib/                (generic system lib)
  - Every dir in LD_LIBRARY_PATH (catches any custom Modal injection)
  Also:
  - Print ldconfig output for libcuda (subprocess.run) for full diagnosis
  - Print /proc/driver/nvidia/version if it exists
  - Print listing of /usr/local/nvidia/ recursively (not just lib64)
"""

import sys
from pathlib import Path

GUARD = "cuinit_sitecustomize_v5"

SITECUSTOMIZE_CONTENT = f'''\
# injected by patch_inject_cuinit_sitecustomize.py ({GUARD})
# Fix LD_LIBRARY_PATH + preload host CUDA driver before any Python imports.
import os as _os, ctypes as _ctypes, glob as _glob, subprocess as _sp

_compat_path = '/usr/local/cuda/compat'
_pid = _os.getpid()

# Step 1: Remove compat from LD_LIBRARY_PATH.
_ld_orig = _os.environ.get('LD_LIBRARY_PATH', '')
_parts = [p for p in _ld_orig.split(':') if p and p != _compat_path]
_ld_fixed = ':'.join(_parts)
_os.environ['LD_LIBRARY_PATH'] = _ld_fixed
print(f'[site_v5 pid={{_pid}}] LD_LIBRARY_PATH: {{_ld_orig!r}} -> {{_ld_fixed!r}}', flush=True)

# Step 2: Broad search for libcuda.so.* (NOT the compat stub).
_SEARCH_PATHS = [
    '/usr/local/nvidia/lib64',
    '/usr/local/nvidia/lib',
    '/usr/lib/x86_64-linux-gnu',
    '/usr/lib64',
    '/usr/lib',
    '/usr/local/lib',
] + [p for p in _parts if p]  # also every dir in LD_LIBRARY_PATH

_host_libcuda = None
for _search_dir in _SEARCH_PATHS:
    _candidates = sorted(_glob.glob(_search_dir + '/libcuda.so*'))
    _real = [
        f for f in _candidates
        if _os.path.isfile(f) and _compat_path not in f
    ]
    if _real:
        _host_libcuda = _real[-1]  # highest version = sort last
        print(f'[site_v5 pid={{_pid}}] found host libcuda in {{_search_dir!r}}: {{_real}}', flush=True)
        break
    elif _os.path.isdir(_search_dir):
        _listing = _os.listdir(_search_dir)
        _cuda_files = [x for x in _listing if 'cuda' in x.lower() or 'nvidia' in x.lower()]
        if _cuda_files:
            print(f'[site_v5 pid={{_pid}}] {{_search_dir!r}} (no libcuda.so*): cuda/nvidia files: {{_cuda_files[:20]}}', flush=True)

# Diagnostic: ldconfig cache for libcuda
try:
    _ldcfg = _sp.run(['ldconfig', '-p'], capture_output=True, text=True, timeout=5)
    _cuda_lines = [l for l in _ldcfg.stdout.splitlines() if 'libcuda' in l]
    print(f'[site_v5 pid={{_pid}}] ldconfig libcuda entries: {{_cuda_lines}}', flush=True)
except Exception as _e:
    print(f'[site_v5 pid={{_pid}}] ldconfig failed: {{_e}}', flush=True)

# Diagnostic: /proc/driver/nvidia/version
try:
    _nvver = open('/proc/driver/nvidia/version').read().strip()
    print(f'[site_v5 pid={{_pid}}] /proc/driver/nvidia/version: {{_nvver!r}}', flush=True)
except Exception:
    print(f'[site_v5 pid={{_pid}}] /proc/driver/nvidia/version: not accessible', flush=True)

# Diagnostic: /usr/local/nvidia/ top-level listing
try:
    _nv_root = '/usr/local/nvidia'
    _nv_listing = []
    for _root, _dirs, _files in _os.walk(_nv_root):
        for _f in _files:
            _nv_listing.append(_os.path.join(_root, _f))
        if len(_nv_listing) > 50:
            _nv_listing.append('...(truncated)...')
            break
    print(f'[site_v5 pid={{_pid}}] /usr/local/nvidia/ files: {{_nv_listing[:50]}}', flush=True)
except Exception as _e:
    print(f'[site_v5 pid={{_pid}}] walk /usr/local/nvidia/ failed: {{_e}}', flush=True)

if _host_libcuda:
    try:
        _lib = _ctypes.CDLL(_host_libcuda)
        _lib.cuInit.restype = _ctypes.c_int
        _ret = _lib.cuInit(0)
        print(f'[site_v5 pid={{_pid}}] cuInit(0) via {{_host_libcuda}} = {{_ret}}', flush=True)
        _preload = _os.environ.get('LD_PRELOAD', '')
        if _host_libcuda not in _preload:
            _os.environ['LD_PRELOAD'] = (_host_libcuda + (':' + _preload if _preload else ''))
            print(f'[site_v5 pid={{_pid}}] LD_PRELOAD -> {{_os.environ["LD_PRELOAD"]!r}}', flush=True)
    except OSError as _e:
        print(f'[site_v5 pid={{_pid}}] CDLL({{_host_libcuda!r}}) failed: {{_e}}', flush=True)
else:
    print(f'[site_v5 pid={{_pid}}] no host libcuda found in any search path', flush=True)

del _os, _ctypes, _glob, _sp, _compat_path, _pid, _ld_orig, _parts, _ld_fixed
del _SEARCH_PATHS, _host_libcuda
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
