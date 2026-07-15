# Building & running on Windows-on-ARM64 (Snapdragon)

The shipped x64 build uses **faster-whisper (ctranslate2) + CUDA**, neither of which
has an ARM64 wheel. On ARM the app runs the transcription engine through
**whisper.cpp via pywhispercpp** instead. The code selects the backend at runtime
(`select_backend()` in `transcriber.py`) — no forked codebase.

Everything below was verified on a Snapdragon Windows 11 ARM64 laptop; the app
transcribes a real clip correctly (native ARM64, CPU).

## The one hard part: there is no pywhispercpp ARM64 wheel
It must be compiled from C++ source, and **whisper.cpp/ggml refuses MSVC on ARM**
(`"MSVC is not supported for ARM, use clang"`). So you need a *clang* toolchain.

> ⚠️ Do NOT judge the environment by `platform.machine()` — on Windows-on-ARM an
> **emulated x64** Python also reports `ARM64`. Verify a *native* interpreter:
> `pip` platform tag must be `win_arm64` and `os.environ['PROCESSOR_ARCHITECTURE']`
> must be `ARM64` (not `AMD64`). `interpreter_is_arm64()` in `transcriber.py`
> encodes this.

## Prerequisites
1. **Native ARM64 Python 3.13** (python.org Windows ARM64, or
   `winget install Python.Python.3.13 --architecture arm64 --scope user`).
   Confirm: `python -c "import sysconfig;print(sysconfig.get_platform())"` → `win-arm64`.
2. A **clang** toolchain. Two options:
   - **rtools45-aarch64** MinGW-w64 clang (LLVM 19, native ARM64) — what this repo's
     build script uses. Zero extra download if already present.
   - A real **MSVC-targeting clang-cl** (VS 2022 "C++ Clang Compiler for Windows" /
     ClangCL toolset, or LLVM `woa64`). Cleaner (no shims/rename) but a large
     download; see "MSVC-clang alternative" below.
3. **ninja** on PATH (the one bundled with VS Build Tools works; so does `pip install ninja`).
4. **MSVC ARM64 build tools + Windows SDK** are only needed for the clang-cl path,
   not for the rtools/MinGW path.

## Steps
```powershell
# 1) Create a NATIVE ARM64 venv
py -3.13-arm64 -m venv .venv-arm64

# 2) Install the ARM-safe deps (native win_arm64 wheels)
.venv-arm64\Scripts\python.exe -m pip install -r requirements-arm.txt

# 3) Build + install pywhispercpp from source (handles shims + the rename)
build_arm\build_pywhispercpp.bat
```
`ctranslate2` / `faster-whisper` will NOT install on ARM (no wheel) — that's expected.

## Why the build needs two shims (rtools/MinGW path)
`build_arm/build_pywhispercpp.bat` sets `CMAKE_GENERATOR=Ninja` (so pywhispercpp's
`setup.py` drops its hardcoded `-A ARM64`, which forces MSVC) and `CC/CXX=clang`,
then applies two build-flag workarounds — **no vendored source is modified**:

1. **`-include build_arm/pt_shim.h`** — rtools' MinGW headers omit the thread-level
   `THREAD_POWER_THROTTLING_STATE` struct that `ggml-cpu.c` uses. The shim supplies it.
2. **`-DPYBIND11_NAMESPACE=pybind11`** — pybind11 wraps its namespace in
   `visibility("hidden")`; clang-mingw rejects `hidden` + `dllexport` together.
   This neutralises the hidden attribute.

Finally the script **renames the built extension**: the MinGW build tags it
`_pywhispercpp.cp313-win_amd64.pyd` even though it's ARM64 machine code (PE
`0xAA64`), so it's renamed to the interpreter's real `EXT_SUFFIX`
(`_pywhispercpp.cp313-win_arm64.pyd`) or it won't import.

## MSVC-clang alternative (no shims, no rename)
With a real MSVC-targeting `clang-cl` + the Windows SDK, none of the above is needed
(the SDK headers define the API; MSVC-mode has no hidden-visibility issue; the ext is
named correctly). Build with the VS generator + ClangCL toolset:
```
set CMAKE_GENERATOR_TOOLSET=ClangCL
call "...\VC\Auxiliary\Build\vcvarsall.bat" arm64
.venv-arm64\Scripts\python.exe -m pip install pywhispercpp --no-cache-dir
```
Prefer this for release builds once the clang-cl toolchain is installed.

## Models
whisper.cpp uses GGML model files (different from faster-whisper). They download on
first use into `whispercpp_models/` (dev) or `%APPDATA%\Transcriber\whispercpp_models`
(installed). CPU/NPU inference is much slower than the x64 GPU path — **default to a
small model on ARM** (e.g. `base.en`, `small.en-q5_1`). Large-v3 across many feeds is
not realistic without a working NPU path.

## What works on ARM (verified on the Snapdragon box)
- Transcription (whisper.cpp), URL feeds, **PC-audio loopback** (`soundcard`),
  Stereo-Mix capture, **"listen to feed"** playback (`sounddevice`), GUI (Tkinter —
  stdlib, no Qt), and **text-to-speech** (WinRT/OneCore, or classic SAPI5).

## Not ported / caveats
- **Per-app capture** (`proc-tap`) does NOT work. Its wheel is `py3-none-any` and
  installs fine, but the compiled `proctap._native` inside has no ARM64 build, so
  captures raise. `proctap_available()` checks for `_native` (not just the package),
  so the "application" source correctly stays hidden rather than failing at capture.
  Everything else (URL feeds, PC-audio loopback, Stereo Mix) is unaffected.
- **ffmpeg** (needed for URL/stream feeds): `imageio-ffmpeg` has no ARM64 wheel, so
  stage a binary the app can find. `find_ffmpeg()` looks for imageio-ffmpeg, then
  **`bin/ffmpeg.exe` next to the app** (or `<user data>/bin`), then PATH; if none
  works the Engine warns once at startup instead of looping on reconnects.
  Get a **native ARM64** build (recommended — an x64 ffmpeg also works, emulated):
  ```powershell
  # BtbN publishes winarm64 builds; grab the *-winarm64-gpl.zip and extract bin/ffmpeg.exe
  winget show BtbN.FFmpeg.GPL --architecture arm64   # shows the current release URL
  # -> extract   <zip>/ffmpeg-*-winarm64-gpl/bin/ffmpeg.exe   to   ./bin/ffmpeg.exe
  ```
  Verify it's really ARM64: the PE machine word should be `0xAA64` (not `0x8664`).
  `bin/` is gitignored (the binary is ~76 MB).
- **Models**: the GUI Model dropdown lists the big x64-oriented models; on ARM prefer a
  small one (`base.en`/`small.en`) — set it in config.json.
- **NPU/GPU**: whisper.cpp here is CPU-only ("no GPU found"); the Snapdragon NPU/Adreno
  is unused. Works fine with small models, but there's headroom.
