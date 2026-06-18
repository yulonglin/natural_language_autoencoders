#!/usr/bin/env python3
"""Inject cuInit(0) into sitecustomize.py so it runs at interpreter startup.

v2: Try the NVIDIA forward-compat libcuda.so.1 at /usr/local/cuda/compat/ first
(absolute path, bypasses LD_LIBRARY_PATH entirely). The standard libcuda.so.1
from the host driver returns Error 803 (CUDA 12.8 > host driver version), but
the compat version is designed to bridge this gap.

Root cause of 803 in scheduler subprocess:
- Container CUDA runtime: 12.8 (pip-installed torch + nvidia-cuda-* packages)
- Host NVIDIA kernel driver: older version (doesn't natively support CUDA 12.8)
- Standard libcuda.so.1 (host driver) → cuInit(0) = 803 (version mismatch)
- NVIDIA forward-compat libcuda.so.1 (/usr/local/cuda/compat/) → cuInit(0) = 0
- Ray workers use the compat path (Ray sets LD_LIBRARY_PATH to include it);
  SGLang scheduler subprocess (mp.Process spawn from HTTP server subprocess.Popen)
  doesn't inherit the right LD_LIBRARY_PATH → gets the old host libcuda.so.1.

Fix: run cuInit(0) via the absolute compat path at sitecustomize.py time (before
any imports), so the CUDA driver context is initialized before import-time probes
(sgl-kernel, flashinfer) can fail with 803 and poison the sticky-error state.
"""
import sys
from pathlib import Path

GUARD = "cuinit_sitecustomize_v2"

SITECUSTOMIZE_CONTENT = f'''\
# injected by patch_inject_cuinit_sitecustomize.py ({GUARD})
# Initialize CUDA driver at interpreter startup via NVIDIA forward-compat library.
# Must run before sgl-kernel / flashinfer CUDA probes to prevent sticky Error 803.
import os as _os, ctypes as _ctypes

_ldpath = _os.environ.get("LD_LIBRARY_PATH", "")
_pid = _os.getpid()
print(f"[cuinit_site pid={{_pid}}] LD_LIBRARY_PATH={{_ldpath!r}}", flush=True)

# Absolute compat paths tried first; fall back to LD_LIBRARY_PATH-based search.
# The forward-compat libcuda.so.1 bridges CUDA 12.8 runtime → older host driver.
_candidates = [
    "/usr/local/cuda/compat/libcuda.so.1",
    "/usr/local/cuda-12.8/compat/libcuda.so.1",
    "/usr/local/cuda-12/compat/libcuda.so.1",
    "libcuda.so.1",
    "libcuda.so",
]

_cuda_init_ok = False
for _libpath in _candidates:
    try:
        _lib = _ctypes.CDLL(_libpath)
        _lib.cuInit.restype = _ctypes.c_int
        _ret = _lib.cuInit(0)
        print(f"[cuinit_site pid={{_pid}}] cuInit(0) via {{_libpath!r}} returned {{_ret}}", flush=True)
        if _ret == 0:
            _cuda_init_ok = True
            break
    except OSError as _e:
        print(f"[cuinit_site pid={{_pid}}] {{_libpath!r}} not found: {{_e}}", flush=True)

if not _cuda_init_ok:
    print(f"[cuinit_site pid={{_pid}}] WARNING: cuInit failed from all candidates", flush=True)

del _os, _ctypes, _ldpath, _pid, _candidates, _cuda_init_ok
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
    # Remove any older version of this injection before prepending v2
    lines = existing.splitlines(keepends=True)
    cleaned = []
    skip = False
    for line in lines:
        if "cuinit_sitecustomize" in line or "cuInit" in line:
            skip = True
        if not skip:
            cleaned.append(line)
        elif line.strip() == "" and skip:
            skip = False
    cleaned_text = "".join(cleaned)
    target.write_text(SITECUSTOMIZE_CONTENT + "\n" + cleaned_text)
    print(f"updated sitecustomize.py with {GUARD} at {target}")
else:
    target.write_text(SITECUSTOMIZE_CONTENT)
    print(f"wrote {GUARD} to {target}")
