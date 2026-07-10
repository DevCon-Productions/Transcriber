# Windows-on-ARM port plan (READ THIS FIRST)

You are a Claude Code session running on a **Windows 11 on ARM (Snapdragon)**
laptop. Your job is to make this app — currently x64 + NVIDIA-only — run on this
ARM machine. This doc is the plan and the findings from the x64 side.

## Rule #0 — protect the shipped x64 release
- Work on the **`arm-support`** branch. Do NOT commit to `master`.
  `git checkout -b arm-support` (or `git checkout arm-support` if it exists).
- `master` is the released x64 v1.0 — leave it untouched. Don't publish releases.
- Keep the codebase SINGLE. The goal is one repo that runs on both x64 and ARM
  via a swappable transcription backend — NOT a forked copy.

## The core blocker (already investigated on PyPI)
Windows ARM64 wheel availability for the compiled dependencies:

| Dependency | ARM64 wheel? | Consequence |
|---|---|---|
| numpy, scipy, pillow, av, sounddevice | YES | fine |
| soundcard, pycaw | pure-python | fine |
| **onnxruntime** | YES | **Piper TTS should still work** |
| **ctranslate2** | **NO** | faster-whisper won't install — must swap engine |
| **proc-tap** | **NO** | per-app capture must be disabled on ARM |

So: the audio stack + GUI + TTS port fine. **The transcription engine
(faster-whisper → ctranslate2) is the one hard blocker.**

## FIRST THING TO DO — prove the environment (make-or-break)
Before writing any port code, verify the engine path is even possible:

1. Confirm **Python 3.13 ARM64**:
   `python -c "import platform,sys; print(platform.machine(), sys.version)"`
   (want `ARM64`). Install from python.org (Windows ARM64 installer) if needed.
2. Create a venv and try the replacement engine:
   `pip install pywhispercpp`
   - If it installs (prebuilt ARM64 wheel) or builds (may need "Visual Studio
     Build Tools" with the **ARM64** C++ toolchain) → **GREEN, proceed.**
   - If it will not build → pivot to **ONNX-Whisper** (onnxruntime has ARM64):
     e.g. `whisper.cpp` alt, or `transformers`+optimum ONNX. Report back.
3. Detect the chip / NPU for later speed tuning:
   `Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores`

## The port plan (once the engine is confirmed)
1. **Abstract the transcription backend.** Today `Engine._make_whisper_model()`
   in `transcriber.py` builds a `faster_whisper.WhisperModel`. Introduce a small
   interface (load + transcribe(audio)->text) with two implementations:
   - `CT2Backend` (existing faster-whisper/ctranslate2) — default on x64.
   - `WhisperCppBackend` (pywhispercpp) — default on ARM.
   Select by platform (`platform.machine()`) or a config key `engine`.
   Keep the transcribe output shape identical so `Transcriber._transcribe` and
   the anti-hallucination filters are unchanged.
2. **CUDA is x64-only.** `ensure_cuda_libraries()` / `add_nvidia_dll_dirs()`
   should no-op on ARM; whisper.cpp uses CPU/NPU, no CUDA.
3. **Disable per-app capture on ARM** (proc-tap missing): guard
   `proctap_available()` — it already returns False if import fails, so the
   "application" source just won't appear. Verify that's graceful.
4. **Models:** whisper.cpp uses GGUF/GGML model files (different from
   faster-whisper). Default to a small/quantized model on ARM (e.g.
   `ggml-small.en-q5_1` or `base.en`) — large-v3 on CPU/NPU will be slow.
   Wire model download for the whisper.cpp format.
5. **Keep dev verifiable:** you're on the real target — after each change, run
   the app (`python -E gui.py`) and a real feed, confirm it transcribes.
6. **Tests:** run the suite (`test_*.py`). The engine-abstraction change may need
   test updates (test_setmodel stubs the model). Keep all green.

## Packaging (later, only after it runs from source)
- PyInstaller build on ARM → an **ARM64** exe. Name the installer distinctly,
  e.g. `Transcriber-ARM64-Setup-1.0.exe`, so it's not confused with the x64 one.
- `Transcriber.spec` / `installer/Transcriber.iss` are the x64 templates to adapt.

## Coordination with the x64 side
- The x64 machine has the original dev context. When you hit a
  cross-cutting design question, note it; the human can relay.
- Commit ARM work to `arm-support` with clear messages. Open a PR to `master`
  only when it's proven and the human approves — don't auto-merge.

## Honest expectations
- **Feasible**, but the engine swap is real work, and CPU/NPU inference will be
  **slower** than the x64 NVIDIA path. "Runs a couple feeds with a small model in
  ~real time" is a realistic target; "large-v3 across 7 feeds" is not, without a
  working NPU path.
- Verify everything on this ARM hardware as you go — that's the whole advantage
  of developing here.
