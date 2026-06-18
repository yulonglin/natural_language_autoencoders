#!/usr/bin/env python3
"""Inject cuInit(0) into sitecustomize.py so it runs at interpreter startup.

Why: import-time CUDA probes (sgl-kernel, flashinfer) in the SGLang scheduler
subprocess call cudaGetDeviceCount() via the CUDA *runtime* API before any of
our patches run. On Modal A100 nodes with CUDA forward-compat, this fails with
Error 803 (driver/runtime version mismatch) and leaves a sticky broken state.

The driver API (cuInit via libcuda.so.1) does NOT have the forward-compat
version check and initialises the CUDA driver context correctly. After cuInit(0)
succeeds, the runtime API's cudaGetDeviceCount() uses the already-initialised
driver context and does not re-check the version.

sitecustomize.py runs at Python interpreter startup, *before* any user imports,
so cuInit(0) fires before sgl-kernel/flashinfer can poison the runtime.

Safe to run in all processes: cuInit(0) is idempotent (multiple calls = no-op).
"""
import sys
from pathlib import Path

SITECUSTOMIZE_CONTENT = '''\
# injected by patch_inject_cuinit_sitecustomize.py
# Initialise CUDA driver API at interpreter startup to prevent Error 803 on
# Modal A100 nodes (CUDA forward-compat: runtime API fails before driver init,
# driver API (cuInit) does NOT check version compat and succeeds).
import os as _os, ctypes as _ctypes, ctypes.util as _ctypes_util

_cuda_init_done = False
for _libname in ('libcuda.so.1', 'libcuda.so', _ctypes_util.find_library('cuda') or ''):
    if not _libname:
        continue
    try:
        _lib = _ctypes.CDLL(_libname)
        _lib.cuInit.restype = _ctypes.c_int
        _ret = _lib.cuInit(0)
        if _os.environ.get('MODAL_803_DEBUG'):
            print(f'[sitecustomize cuInit] {_libname} returned {_ret}', flush=True)
        _cuda_init_done = True
        break
    except OSError:
        continue

del _os, _ctypes, _ctypes_util, _libname, _lib, _ret, _cuda_init_done
'''

# Find the site-packages directory for the current Python
import site
candidates = site.getsitepackages() if hasattr(site, "getsitepackages") else []
# Try standard path as fallback
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
    if "cuInit" in existing:
        print(f"sitecustomize.py already has cuInit injection at {target}")
        sys.exit(0)
    # Prepend to existing sitecustomize
    target.write_text(SITECUSTOMIZE_CONTENT + "\n" + existing)
    print(f"prepended cuInit injection to existing {target}")
else:
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"wrote cuInit injection to {target}")
