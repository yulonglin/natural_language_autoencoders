#!/usr/bin/env python3
"""Patch SGLang model_runner.py to fix CUDA Error 803 on Modal A100 nodes.

v4: Try the NVIDIA forward-compat libcuda.so.1 (/usr/local/cuda/compat/) FIRST,
using an absolute path (bypasses LD_LIBRARY_PATH). The standard libcuda.so.1 from
the host driver returns Error 803 (CUDA 12.8 > host driver version); the compat
version bridges this gap. Also prints LD_LIBRARY_PATH for diagnosis.

Root cause (two-layer problem):

Layer 1 — CUDA_VISIBLE_DEVICES not set before scheduler spawn:
  Fix: patch_sglang_device_reindex.py (patches common.py, applied earlier).

Layer 2 — cuInit(0) itself returns 803 in scheduler subprocess:
  Standard libcuda.so.1 (host driver, too old for CUDA 12.8) → cuInit = 803.
  NVIDIA compat libcuda.so.1 at /usr/local/cuda/compat/ → cuInit = 0 (expected).
  Ray workers use the compat path; scheduler subprocess doesn't → Error 803.

Fix (primary): sitecustomize.py injection calls cuInit(0) via compat path at
  interpreter startup, before any import-time probes can set the sticky 803 state.
  This patch is a belt-and-suspenders fallback in case sitecustomize.py doesn't
  find the compat path or runs too late.
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
GUARD = "modal_cuda_803_fix_v4"

if GUARD in code:
    print("model_runner.py already patched (v4)")
    sys.exit(0)

# Remove any older version of this patch before inserting v4
for old_guard in ("modal_cuda_803_fix_v3", "modal_cuda_803_fix_v2", "modal_cuda_803_fix_v1"):
    if old_guard in code:
        # Find the inserted block: it starts just before TARGET's line and ends AT TARGET's line.
        # Simplest: strip all lines containing the old guard comment and the inserted block.
        lines = code.splitlines(keepends=True)
        new_lines = []
        in_block = False
        for line in lines:
            if old_guard in line:
                in_block = True
            if not in_block:
                new_lines.append(line)
            if in_block and TARGET in line:
                # This line IS the target we inject before; keep it and stop stripping.
                new_lines.append(line)
                in_block = False
        code = "".join(new_lines)
        print(f"stripped old patch ({old_guard}) from model_runner.py")
        break

if TARGET not in code:
    print("target not found in model_runner.py")
    sys.exit(1)

idx = code.index(TARGET)
line_start = code.rfind("\n", 0, idx) + 1
indent = code[line_start:idx]

insert = (
    indent + f"# {GUARD}: init NVIDIA compat CUDA driver before any runtime call.\n"
    + indent + "import ctypes as _ctypes, ctypes.util as _ctypes_util, os as _os\n"
    + indent + "print(f'[modal_803_fix v4] LD_LIBRARY_PATH={_os.environ.get(\"LD_LIBRARY_PATH\")!r}')\n"
    + indent + "print(f'[modal_803_fix v4] CUDA_VISIBLE_DEVICES={_os.environ.get(\"CUDA_VISIBLE_DEVICES\")!r} gpu_id={self.gpu_id}')\n"
    + indent + "_cu_init_ok = False\n"
    # Absolute compat paths first (bypass LD_LIBRARY_PATH), then LD_LIBRARY_PATH-based
    + indent + "for _libname in ('/usr/local/cuda/compat/libcuda.so.1', '/usr/local/cuda-12.8/compat/libcuda.so.1', 'libcuda.so.1', 'libcuda.so', _ctypes_util.find_library('cuda')):\n"
    + indent + "    if not _libname:\n"
    + indent + "        continue\n"
    + indent + "    try:\n"
    + indent + "        _libcuda = _ctypes.CDLL(_libname)\n"
    + indent + "        _libcuda.cuInit.restype = _ctypes.c_int\n"
    + indent + "        _cu_init_ret = _libcuda.cuInit(0)\n"
    + indent + "        print(f'[modal_803_fix v4] cuInit(0) via {_libname!r} returned {_cu_init_ret}')\n"
    + indent + "        if _cu_init_ret == 0:\n"
    + indent + "            _cu_init_ok = True\n"
    + indent + "            break\n"
    + indent + "    except OSError as _e:\n"
    + indent + "        print(f'[modal_803_fix v4] {_libname!r} not found: {_e}')\n"
    + indent + "        continue\n"
    + indent + "for _rtlib in ('libcudart.so.12', 'libcudart.so'):\n"
    + indent + "    try:\n"
    + indent + "        _rt = _ctypes.CDLL(_rtlib)\n"
    + indent + "        _rt.cudaGetLastError.restype = _ctypes.c_int\n"
    + indent + "        _rt_err = _rt.cudaGetLastError()\n"
    + indent + "        print(f'[modal_803_fix v4] cudaGetLastError via {_rtlib} returned {_rt_err}')\n"
    + indent + "        break\n"
    + indent + "    except OSError:\n"
    + indent + "        continue\n"
    # Layer 1 fallback: set CUDA_VISIBLE_DEVICES if common.py patch didn't run.
    + indent + "if _os.environ.get('CUDA_VISIBLE_DEVICES') is None and self.gpu_id >= 0:\n"
    + indent + "    _os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)\n"
    + indent + "    self.gpu_id = 0\n"
    + indent + "    print(f'[modal_803_fix v4] set CUDA_VISIBLE_DEVICES (fallback)')\n"
)

f.write_text(code[:line_start] + insert + code[line_start:])
print("patched model_runner.py (v4): compat cuInit + cudaGetLastError + LD_LIBRARY_PATH debug")
