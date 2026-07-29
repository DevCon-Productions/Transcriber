# Bringing 2.0 to the x64 build

Written for whoever picks up the x64 side. Read this before touching code — the
framing matters more than the steps.

## This is a merge, not a port

Everything on `arm-support` is architecture-neutral. It uses the Python standard
library plus the ffmpeg that both builds already ship. Nothing added in 2.0
touches CUDA, ctranslate2, or whisper.cpp.

That was a deliberate constraint throughout, and it's why specific choices look
odd in isolation:

- The **PDF writer is hand-rolled** against the PDF 1.4 spec instead of using
  reportlab, because the ARM build installs from a hand-curated wheel list and
  every new dependency is a wheel that might not exist for `win_arm64`.
- **Transcript bundles use `zipfile`** (stdlib) rather than a binary container.
- **MP3 and Opus encoding go through the bundled ffmpeg.** `libmp3lame` and
  `libopus` were confirmed present in the x64 `imageio-ffmpeg` build, so no
  action is needed there.

So the expected path is: merge the branch, run the suite, build the installer.
Do **not** reimplement anything.

`arm-support` is 39 commits ahead of `master`; 18 of those are the 2.0 feature
work. 26 files, ~8,400 insertions.

## Decide first: how much to merge

The 21 older commits underneath are ARM packaging — a separate Inno installer, a
distinct `%APPDATA%` data dir, an arch-aware updater asset picker. They are
harmless on `master` but conceptually belong to the ARM build.

- **Merge everything** — simplest, keeps the branches from diverging further.
- **Cherry-pick the 18 feature commits** — keeps `master` free of ARM packaging.

This is the repo owner's call. Ask before assuming.

## Verify on x64, highest risk first

### 1. `proctap_available()` — the one with real downside

It was rewritten to answer **from disk** and never import `proctap`, because on
ARM64 importing it and then enumerating speakers through `soundcard` corrupts the
heap and kills the process (`0xC0000374`).

On x64, per-app capture **actually works**, so this function must still return
`True` there. It now looks for a `_native*.pyd` beside `proctap/__init__.py`.

**Test on a machine with `proc-tap` installed:**

```python
import transcriber as core
print(core.proctap_available())   # must be True on x64 with proc-tap installed
```

If it returns `False`, the "application" source silently disappears from the
add-feed dialog. This is the only change in 2.0 that can regress an x64 feature.

### 2. Audio device enumeration is now lazy

`AddStreamDialog` no longer calls `list_output_devices()` when it opens; it waits
until the user selects the "pc audio" source. On ARM the eager call was crashing
the app outright. On x64 it should be a pure improvement (the dialog stops
querying WASAPI every time it opens), but confirm the speaker picker still
populates when "pc audio" is chosen.

`ChangeDeviceDialog` and `probe_output_level` still enumerate eagerly, by design —
they exist to pick a device. The durable fix is isolating enumeration in a
subprocess so a crash costs a device list rather than the app; that needs a
frozen-build entry point and was left undone.

### 3. `sounddevice` is missing from `requirements.txt`

`requirements-arm.txt` lists it explicitly; the x64 list never has. On a clean
x64 install that means **Listen, TTS, and now clip playback are all unavailable**.
It predates 2.0, but clip playback raises the stakes. Worth adding.

### 4. Config seeding

`config.example.json` gained a `clips` block. Existing installs without it fall
back to `CLIP_DEFAULTS` (recording **off**), so no migration is needed.

## The v2.0 release already exists

**Do not create a new tag.** `v2.0` is published, marked Latest, carrying only
`Transcriber-ARM64-Setup-2.0-arm64.exe`. Add the x64 installer as an asset to
**that same release**:

```powershell
gh release upload v2.0 installer\Output\Transcriber-Setup-2.0.exe
```

Then edit the notes to drop the paragraph saying the x64 installer isn't
published yet.

### Why one release carries both

`check_for_app_update()` queries `/releases/latest`, which **excludes
pre-releases**. The old scheme — `v1.4` for x64 plus a `v1.4-arm64`
pre-release — meant the ARM updater never saw an ARM build at all: it got the
x64 release, found no arm64 asset, and reported "up to date" forever. 2.0 fixes
that by putting both architectures on one non-prerelease tag, which is what
`_pick_installer_asset()` was written to expect. Keep it that way.

Until the x64 asset is uploaded, an x64 user checking for updates is told 2.0
exists and no download starts. That path was verified; it does not error.

## Build steps (x64)

Per `BUILD_AND_RELEASE.md`:

```powershell
.venv\Scripts\python.exe -E -m PyInstaller Transcriber.spec --noconfirm
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\Transcriber.iss
```

Bump `APP_VERSION` in `gui.py` to `2.0` (the ARM branch uses `2.0-arm64`), and
`MyAppVersion` in `installer\Transcriber.iss`. `test_gui_smoke.py` asserts the
version string — update it in the same commit.

Before publishing, confirm the built exe is genuinely x64 and launches, loads a
model, and writes to `%APPDATA%\Transcriber` (not the ARM dir).

## Tests

Plain scripts, not pytest. Run each directly:

```powershell
.venv\Scripts\python.exe test_gui_smoke.py
```

17 files. `test_pcaudio.py` needs `sounddevice` (see item 3). The GUI smoke test
is the broad one — ~200 checks covering clips, bundles, MP3 export, retention,
service profiles, column groups and row selection.

Two suites cannot be meaningfully verified on x64: the ARM crash fixes. That's
expected — just don't "fix" them back.

## One behavioural note for x64

The ARM build ships a small model out of necessity. On a real corpus of ATC
audio, `base.en` produced confident nonsense where `small.en` recovered the
operationally meaningful tokens — runway identifiers, wind, "cleared to land".
x64 has `large-v3` on CUDA available, so ATC transcription should be markedly
better there. The model is a **global** config setting, not per-feed; per-feed
model selection is an obvious extension if it's wanted, and the per-feed plumbing
(service profiles, prompt overrides) already exists to hang it on.
