#!/usr/bin/env python3
"""Inject LD_LIBRARY_PATH fix + host-driver cuInit into sitecustomize.py.

v3 — ROOT CAUSE FIX for Modal Error 803 (cudaErrorSystemDriverMismatch)
======================================================================

Diagnosis (from run bc4gvqrl8, 2026-06-19):
  LD_LIBRARY_PATH = '/usr/local/cuda/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:...'
  - Modal's NVIDIA container toolkit injects /usr/local/cuda/compat FIRST.
  - /usr/local/cuda/compat/libcuda.so.1 = compat library baked into the CUDA 12.8 image.
    This version doesn't match Modal's kernel module → cuInit(0) = 803.
  - /usr/local/nvidia/lib64/libcuda.so.1 = host driver injected at runtime by the container
    toolkit. Matches the kernel module → cuInit(0) = 0 (or 100 = no-device on build machines).

v2 mistake: v2 loaded the compat libcuda via ctypes and called cuInit → got 803 and cached
the failure as the canonical 'libcuda.so.1' in the dynamic linker. All subsequent dlopen(
'libcuda.so.1') calls returned the compat handle with poisoned 803 state. NLAFSDPActors
worked because PyTorch's CUDA init goes through cuDevicePrimaryCtxRetain (driver API) without
first calling cudaGetDeviceCount → no cuInit poisoning. SGLangEngine called torch.cuda.
is_available() → cudaGetDeviceCount() → cuInit(0) via compat → cached 803 → set_device fails.

v3 fix (this file):
  1. In sitecustomize.py (runs at Python interpreter startup, before any imports):
     a. Remove /usr/local/cuda/compat from LD_LIBRARY_PATH.
     b. Move /usr/local/nvidia/lib64 to the front.
     c. Load /usr/local/nvidia/lib64/libcuda.so.1 via ctypes → registers it as canonical
        'libcuda.so.1' in the dynamic linker cache before torch loads any CUDA library.
     d. Call cuInit(0) on the host driver → 0 on GPU worker, 100 on build machine (expected).
  2. Subsequent 'import torch' → torch loads libcuda.so.1 from linker cache → host driver →
     cuInit returns 0 → no 803 state cached.
  3. Scheduler subprocess (spawned by SGLang) inherits the fixed LD_LIBRARY_PATH and runs
     sitecustomize again → same fix applies.
  NEVER load /usr/local/cuda/compat/libcuda.so.1 via ctypes — doing so poisons the linker.
"""

import sys
from pathlib import Path

GUARD = "cuinit_sitecustomize_v3"

SITECUSTOMIZE_CONTENT = f'''\
# injected by patch_inject_cuinit_sitecustomize.py ({GUARD})
# Fix LD_LIBRARY_PATH at interpreter startup: move /usr/local/nvidia/lib64 (Modal's
# injected host CUDA driver) before /usr/local/cuda/compat (compat library in container
# image that mismatches Modal's kernel module and returns cuInit=803).
# Then load the host driver's libcuda.so.1 to register it as the canonical 'libcuda.so.1'
# in the dynamic linker cache, so that 'import torch' picks up the correct library.
import os as _os, ctypes as _ctypes

_nvidia_lib64 = '/usr/local/nvidia/lib64'
_compat_path  = '/usr/local/cuda/compat'
_ld_orig = _os.environ.get('LD_LIBRARY_PATH', '')
_parts = [p for p in _ld_orig.split(':') if p and p != _compat_path]
if _nvidia_lib64 in _parts:
    _parts.remove(_nvidia_lib64)
_parts.insert(0, _nvidia_lib64)
_ld_fixed = ':'.join(_parts)
_os.environ['LD_LIBRARY_PATH'] = _ld_fixed
print(f'[cuinit_site_v3 pid={{_os.getpid()}}] LD_LIBRARY_PATH: {{_ld_orig!r}} -> {{_ld_fixed!r}}', flush=True)

# Load host driver's libcuda.so.1 to register it as canonical before torch imports.
# DO NOT load /usr/local/cuda/compat/libcuda.so.1 — that poisons the linker cache.
_host_libcuda = _nvidia_lib64 + '/libcuda.so.1'
try:
    _libcuda = _ctypes.CDLL(_host_libcuda)
    _libcuda.cuInit.restype = _ctypes.c_int
    _ret = _libcuda.cuInit(0)
    # 0 = success (GPU worker), 100 = CUDA_ERROR_NO_DEVICE (build machine, no GPU), else = problem
    print(f'[cuinit_site_v3 pid={{_os.getpid()}}] cuInit(0) via host driver = {{_ret}}', flush=True)
except OSError as _e:
    print(f'[cuinit_site_v3 pid={{_os.getpid()}}] host driver not found at {{_host_libcuda!r}}: {{_e}}', flush=True)

del _nvidia_lib64, _compat_path, _ld_orig, _parts, _ld_fixed, _host_libcuda, _os, _ctypes
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
    # Overwrite entirely — the sitecustomize is only written by our patches.
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"replaced sitecustomize.py with {GUARD} at {target} (was: {len(existing)} chars)")
else:
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"wrote {GUARD} to {target}")
