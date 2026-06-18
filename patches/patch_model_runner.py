#!/usr/bin/env python3
"""Patch SGLang model_runner.py to fix CUDA Error 803 on Modal A100 nodes.

Root cause (two-layer problem):

Layer 1 — CUDA_VISIBLE_DEVICES not set before scheduler spawn:
  Fix: patch_sglang_device_reindex.py (patches common.py, applied earlier).

Layer 2 — CUDA runtime poisoned by import-time probes (THIS PATCH):
  Even with CUDA_VISIBLE_DEVICES correctly set to a single GPU, forward-compat
  CUDA on Modal A100 nodes causes cudaGetDeviceCount() to fail with Error 803.

Fix: before set_device(), call cuInit(0) via the real CUDA driver API (libcuda.so.1,
  NOT the stub libcuda.so) to initialize the driver context. Then call
  cudaGetLastError() to clear any sticky 803 from prior import probes.
  With the driver context properly initialized, the subsequent runtime API call
  set_device(0) → _cuda_init() → cudaGetDeviceCount() should succeed.
"""
import sys
from pathlib import Path

target_path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "/root/sglang/python/sglang/srt/model_executor/model_runner.py"
)
f = Path(target_path)

if not f.exists():
    print(f"model_runner.py not found at {target_path}, skipping")
    sys.exit(0)

code = f.read_text()
TARGET = "torch.get_device_module(self.device).set_device(self.gpu_id)"
GUARD = "modal_cuda_803_fix_v3"

if GUARD in code:
    print("model_runner.py already patched (v3)")
    sys.exit(0)

if TARGET not in code:
    print("target not found in model_runner.py")
    sys.exit(1)

idx = code.index(TARGET)
line_start = code.rfind("\n", 0, idx) + 1
indent = code[line_start:idx]

insert = (
    indent + "# modal_cuda_803_fix_v3: init real CUDA driver before any runtime call.\n"
    + indent + "# libcuda.so may be a stub on Modal; try libcuda.so.1 (actual versioned driver).\n"
    + indent + "# cuInit(0) via the real driver initialises the CUDA driver context without\n"
    + indent + "# the forward-compat version check that cudaGetDeviceCount() performs.\n"
    + indent + "# cudaGetLastError() clears any sticky 803 from prior import-time probes.\n"
    + indent + "import ctypes as _ctypes, ctypes.util as _ctypes_util, os as _os\n"
    + indent + "_cu_init_ret = None\n"
    + indent + "for _libname in ('libcuda.so.1', 'libcuda.so', _ctypes_util.find_library('cuda')):\n"
    + indent + "    if not _libname:\n"
    + indent + "        continue\n"
    + indent + "    try:\n"
    + indent + "        _libcuda = _ctypes.CDLL(_libname)\n"
    + indent + "        _libcuda.cuInit.restype = _ctypes.c_int\n"
    + indent + "        _cu_init_ret = _libcuda.cuInit(0)\n"
    + indent + "        print(f'[modal_803_fix] cuInit(0) via {_libname} returned {_cu_init_ret}')\n"
    + indent + "        break\n"
    + indent + "    except OSError as _e:\n"
    + indent + "        print(f'[modal_803_fix] {_libname} not found: {_e}')\n"
    + indent + "        continue\n"
    + indent + "for _rtlib in ('libcudart.so.12', 'libcudart.so'):\n"
    + indent + "    try:\n"
    + indent + "        _rt = _ctypes.CDLL(_rtlib)\n"
    + indent + "        _rt.cudaGetLastError.restype = _ctypes.c_int\n"
    + indent + "        _rt_err = _rt.cudaGetLastError()\n"
    + indent + "        print(f'[modal_803_fix] cudaGetLastError via {_rtlib} returned {_rt_err}')\n"
    + indent + "        break\n"
    + indent + "    except OSError:\n"
    + indent + "        continue\n"
    + indent + "print(f'[modal_803_fix] about to set_device({self.gpu_id}) CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r}')\n"
    + indent + "# Layer 1 fallback: set CUDA_VISIBLE_DEVICES if common.py patch didn't run.\n"
    + indent + "if _os.environ.get('CUDA_VISIBLE_DEVICES') is None and self.gpu_id >= 0:\n"
    + indent + "    _os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)\n"
    + indent + "    self.gpu_id = 0  # remap: single visible GPU is always device 0\n"
)

f.write_text(code[:line_start] + insert + code[line_start:])
print("patched model_runner.py: cuInit(libcuda.so.1) + cudaGetLastError() + debug output")
