# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for Transcriber (DevCon Productions).

Build:   .venv\\Scripts\\python.exe -E -m PyInstaller Transcriber.spec --noconfirm
Output:  dist\\Transcriber\\Transcriber.exe   (one-folder build)

Notes:
- One-FOLDER build (not one-file): the app pulls in large CUDA/ML libraries;
  one-folder starts faster and is friendlier for an installer.
- Whisper models are NOT bundled (they download to the HF cache on first run,
  ~3 GB). Piper TTS voices in tts_voices/ ARE bundled if present.
- Secrets (credentials.json, config.json) are intentionally NOT bundled.
"""
import os
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, collect_data_files

block_cipher = None

# --- data files to ship alongside the exe --------------------------------
datas = []
for asset in ("OfficialLogo.png", "OfficialTaskbarIcon.png", "Developer.png",
              "app.ico", "config.example.json", "credentials.example.json",
              "README.md", "APP_ROUTING.md", "MULTICHANNEL.md"):
    if os.path.exists(asset):
        datas.append((asset, "."))

# Bundle Piper voice models if the folder exists (each ~60 MB).
if os.path.isdir("tts_voices"):
    datas.append(("tts_voices", "tts_voices"))

# Piper ships an espeak-ng data dir + onnx pieces it needs at runtime.
datas += collect_data_files("piper")
datas += collect_data_files("piper_phonemize", include_py_files=False) \
    if os.path.exists("./.venv/Lib/site-packages/piper_phonemize") else []

# --- native binaries -----------------------------------------------------
# NOTE: the large NVIDIA CUDA libraries (~1.9 GB) are deliberately NOT bundled.
# The app downloads them into %APPDATA%\Transcriber\cuda on first run
# (ensure_cuda_libraries), keeping the installer small enough for GitHub.
binaries = []
for pkg in ("ctranslate2", "onnxruntime", "soundcard", "proctap"):
    try:
        binaries += collect_dynamic_libs(pkg)
    except Exception:
        pass

# Belt-and-suspenders: drop anything from the nvidia packages if PyInstaller
# pulled them in transitively. Entries are (src_path, dest_dir) tuples.
binaries = [entry for entry in binaries
            if "nvidia" not in str(entry[0]).lower().replace("\\", "/")
            and "nvidia" not in str(entry[1]).lower().replace("\\", "/")]

# --- hidden imports (dynamically-loaded modules PyInstaller can miss) -----
hiddenimports = []
for pkg in ("piper", "onnxruntime", "soundcard", "proctap", "pycaw",
            "comtypes", "sounddevice", "faster_whisper", "ctranslate2",
            "PIL", "PIL.ImageTk", "numpy", "scipy"):
    try:
        hiddenimports += collect_submodules(pkg)
    except Exception:
        hiddenimports.append(pkg)

a = Analysis(
    ["gui.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter.test", "test", "pytest",
              # CUDA libs are downloaded on first run, not bundled (keeps the
              # installer small). See ensure_cuda_libraries().
              "nvidia", "nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_nvrtc",
              "nvidia.cuda_runtime", "nvidia.cufft", "nvidia.curand"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
