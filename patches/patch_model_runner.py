#!/usr/bin/env python3
"""Patch SGLang model_runner.py to fix CUDA Error 803 on Modal A100 nodes.

Root cause (two-layer problem):

Layer 1 — CUDA_VISIBLE_DEVICES not set before scheduler spawn:
  Miles uses launch_server_process (HTTP server mode, subprocess.Popen) rather
  than SGLang's Engine class (mp.Process). SGLang's maybe_reindex_device_id
  only wraps mp.Process.start(), so without the common.py patch it never runs.
  Fix: patch_sglang_device_reindex.py (patches common.py, applied earlier).

Layer 2 — CUDA runtime poisoned by import-time probes (THIS PATCH):
  Even with CUDA_VISIBLE_DEVICES correctly set to a single GPU, the scheduler
  subprocess's Python import chain (sgl-kernel, flashinfer) probes CUDA during
  import via torch.cuda.is_available() → cudaGetDeviceCount(). On Modal A100
  nodes in CUDA forward-compat mode, cudaGetDeviceCount() fails with Error 803
  and leaves the CUDA runtime in a sticky broken state. All subsequent runtime
  API calls (including the set_device() below) then inherit the 803 error.

Fix: before set_device(), call cuInit(0) via the CUDA driver API (libcuda.so).
  The driver API bypasses the forward-compat version check and initializes the
  CUDA driver. Then call cudaGetLastError() to clear any sticky 803 from prior
  import-time probes. With the driver properly initialized, the subsequent
  runtime API call set_device(0) → _cuda_init() → cudaGetDeviceCount() succeeds.

Also: fall back to setting CUDA_VISIBLE_DEVICES here if common.py patch wasn't
applied (defensive — common.py patch should handle this, but belt-and-suspenders).
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
GUARD = "cuInit(0)  # driver API: init before runtime calls"

if GUARD in code:
    print("model_runner.py already patched")
    sys.exit(0)

if TARGET not in code:
    print("target not found in model_runner.py")
    sys.exit(1)

idx = code.index(TARGET)
line_start = code.rfind("\n", 0, idx) + 1
indent = code[line_start:idx]

insert = (
    indent + "# Modal CUDA Error 803 fix (layer 2): init driver API before any runtime call.\n"
    + indent + "# Forward-compat CUDA on Modal: cudaGetDeviceCount() in import-time probes\n"
    + indent + "# (sgl-kernel, flashinfer) leaves the runtime in a sticky 803 state.\n"
    + indent + "# cuInit(0) via libcuda.so initialises the driver without a version check;\n"
    + indent + "# cudaGetLastError() clears the sticky 803 so set_device() can succeed.\n"
    + indent + "import ctypes as _ctypes, os as _os\n"
    + indent + "try:\n"
    + indent + "    _ctypes.CDLL('libcuda.so').cuInit(0)  # driver API: init before runtime calls\n"
    + indent + "except Exception:\n"
    + indent + "    pass\n"
    + indent + "try:\n"
    + indent + "    for _lib in ('libcudart.so.12', 'libcudart.so'):\n"
    + indent + "        try:\n"
    + indent + "            _ctypes.CDLL(_lib).cudaGetLastError()  # clear sticky 803\n"
    + indent + "            break\n"
    + indent + "        except OSError:\n"
    + indent + "            pass\n"
    + indent + "except Exception:\n"
    + indent + "    pass\n"
    + indent + "# Layer 1 fallback: set CUDA_VISIBLE_DEVICES if common.py patch didn't run.\n"
    + indent + "if _os.environ.get('CUDA_VISIBLE_DEVICES') is None and self.gpu_id >= 0:\n"
    + indent + "    _os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)\n"
    + indent + "    self.gpu_id = 0  # remap: single visible GPU is always device 0\n"
)

f.write_text(code[:line_start] + insert + code[line_start:])
print("patched model_runner.py: cuInit(0) + cudaGetLastError() before set_device()")
