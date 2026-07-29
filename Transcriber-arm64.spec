# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Transcriber — Windows-on-ARM64 (native).

Build:   .venv-arm64\\Scripts\\python.exe -E -m PyInstaller Transcriber-arm64.spec --noconfirm
Output:  dist\\Transcriber\\Transcriber.exe   (one-folder build)

ARM differs from the x64 spec (Transcriber.spec):
- Transcription engine is **whisper.cpp** (`pywhispercpp`), NOT faster-whisper/
  ctranslate2/CUDA — none of which have ARM64 wheels, so they're excluded.
- TTS is **WinRT/OneCore** (`winrt-*`) with a classic SAPI5 (`comtypes`) fallback,
  NOT Piper (its espeak-ng phonemizer has no ARM64 build).
- Bundles the native ARM64 `bin\\ffmpeg.exe` (imageio-ffmpeg has no ARM64 wheel).
- Seeds a small CPU model on first run via `config.example.arm.json` (the shared
  config.example.json defaults to large-v3/CUDA -> a 3 GB download + unusable on CPU).

Whisper.cpp GGUF models are NOT bundled (base.en ~148 MB downloads on first use).
Secrets (credentials.json, config.json) are intentionally NOT bundled.
"""
import os
from PyInstaller.utils.hooks import (collect_all, collect_dynamic_libs,
                                     collect_submodules)

datas, binaries, hiddenimports = [], [], []

# --- app assets + the ARM fresh-install config seed ----------------------
for asset in ("OfficialLogo.png", "OfficialTaskbarIcon.png", "Developer.png",
              "app.ico", "config.example.arm.json", "credentials.example.json",
              "README.md", "APP_ROUTING.md", "MULTICHANNEL.md"):
    if os.path.exists(asset):
        datas.append((asset, "."))

# --- native ARM64 ffmpeg -> bin/ (find_ffmpeg() looks in HERE/bin) --------
if os.path.exists(os.path.join("bin", "ffmpeg.exe")):
    binaries.append((os.path.join("bin", "ffmpeg.exe"), "bin"))

# --- compiled / runtime-loaded deps: grab datas+binaries+hiddenimports ----
# collect_all handles the C-extensions (pywhispercpp), the WinRT projection
# (namespace packages with .pyds), onnxruntime's native libs, comtypes, etc.
for pkg in ("pywhispercpp", "comtypes",
            "soundcard", "sounddevice", "pycaw", "PIL",
            "winrt",
            "winrt.windows.media.speechsynthesis",
            "winrt.windows.storage.streams",
            "winrt.windows.foundation",
            "winrt.windows.foundation.collections"):
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# Belt-and-suspenders for the whisper.cpp C-extension + a couple of modules
# PyInstaller can miss.
try:
    binaries += collect_dynamic_libs("pywhispercpp")
except Exception:
    pass
try:
    hiddenimports += collect_submodules("winrt")
except Exception:
    pass
hiddenimports += ["_pywhispercpp", "PIL.ImageTk", "comtypes.gen"]

a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter.test", "test", "pytest",
        # x64-only transcription stack — no ARM64 wheels, never used on ARM.
        "faster_whisper", "ctranslate2", "torch", "piper", "piper_phonemize",
        "nvidia", "nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc",
        "nvidia.cuda_runtime", "nvidia.cufft", "nvidia.curand",
        # Not used by the ARM app: onnxruntime (was for Piper), scipy + proctap
        # (per-app capture has no ARM64 native ext), onnx. Excluded to avoid a
        # heavy/broken analysis and shrink the build.
        "onnxruntime", "onnx", "scipy", "proctap",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Transcriber",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,                 # windowed app (no console)
    icon="app.ico",
    version_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Transcriber",
)
