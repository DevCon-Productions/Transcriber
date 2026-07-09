"""Quick GPU smoke test for faster-whisper on this machine.

Verifies that ctranslate2 can find the cuBLAS/cuDNN DLLs (installed as pip
packages under .venv/Lib/site-packages/nvidia/*/bin) and run Whisper on the GPU.
Run with the isolated flag so the global PYTHONPATH does not leak in:
    .venv/Scripts/python.exe -E gpu_smoketest.py
"""
import os
import sys
import glob
import time

# --- Make the pip-installed NVIDIA DLLs discoverable by ctranslate2 ---
def add_nvidia_dll_dirs():
    site = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "nvidia")
    site = os.path.abspath(site)
    found = []
    for bindir in glob.glob(os.path.join(site, "*", "bin")):
        os.add_dll_directory(bindir)
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
        found.append(bindir)
    return found

found = add_nvidia_dll_dirs()
print("Added NVIDIA DLL dirs:")
for f in found:
    print("   ", f)

import numpy as np
from faster_whisper import WhisperModel

print("\nLoading 'tiny' model on CUDA (float16)...")
t0 = time.time()
model = WhisperModel("tiny", device="cuda", compute_type="float16")
print(f"  model loaded in {time.time()-t0:.1f}s")

# 3 seconds of silence -> just exercise the GPU path end to end.
audio = np.zeros(16000 * 3, dtype=np.float32)
t0 = time.time()
segments, info = model.transcribe(audio, language="en")
segs = list(segments)
print(f"  transcribe ran on GPU in {time.time()-t0:.2f}s; segments={len(segs)}")
print("\nGPU SMOKE TEST: PASS")
