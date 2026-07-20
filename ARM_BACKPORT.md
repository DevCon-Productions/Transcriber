# Backporting x64 features to the ARM build (`arm-support`)

Handoff from the x64/master session (2026-07-18). This lists everything that
landed on `master` **after the ARM fork** and how to bring it into the ARM build.

## Fork point & commits to port

The ARM branch forked at **`31f08b6`** ("README: x64 requirements"). Six commits
landed on `master` since:

| Commit | What | Port? |
|--------|------|-------|
| `c9d9da0` | TTS: persistent spoken-line highlight (green→blue) | ✅ Yes |
| `106e31b` | Clickable address → Google Maps links | ✅ Yes |
| `f31eda9` | Scroll-position preservation (v1.1) | ✅ Yes (gui.py hunks only) |
| `dc1c6f7` | Bundle faster-whisper Silero VAD in installer | ❌ **x64-only — do NOT port literally** |
| `e28430e` | In-app Broadcastify login dialog | ✅ **Yes — highest value** |
| `958735b` | v1.2 version bump | ❌ Keep your own ARM version |

## Before you start

1. **Commit and push your ARM work first** (`git push origin arm-support`) so a
   merge/cherry-pick is recoverable and so the x64 session can see your actual
   divergence next time.
2. `git fetch origin master`
3. Prefer `git cherry-pick <hash>` for the clean features; hand-port where your
   backend refactor has touched the same functions. All feature code is
   backend-agnostic except where noted.

---

## PORT THESE (feature parity)

### 1. Broadcastify login dialog — do this first, highest value
Commit **`e28430e`**. Completely backend-agnostic (auth is about *stream URLs*,
not transcription). **It fixes the exact "fresh install → feeds silently drop"
bug ARM installs will also hit**: the installer seeds a placeholder
`credentials.json`, and before this change the app tried to authenticate with the
literal placeholder text.

- **transcriber.py**: `_PLACEHOLDER_CREDS` + `_clean_cred()` (placeholder/blank
  values now ignored in `load_credentials`), `save_credentials()`,
  `credentials_configured()`, `is_broadcastify_stream()`, and
  `Engine.apply_credentials(user, pw, active_streams)` (saves → rebuilds the auth
  header → hot-restarts only the running Broadcastify feeds; no app restart).
- **gui.py**: `BroadcastifyLoginDialog` class, `_open_login()`,
  `_maybe_prompt_login()`, the **Streams → "Broadcastify login…"** menu item, and
  the call to `_maybe_prompt_login()` from the `"ready"` event handler.
- **tests**: `test_credentials.py` (20 checks, backend-independent — should pass
  as-is on ARM).
- **ARM caveat**: none. `apply_credentials` restarts streams via
  `remove_stream`/`add_stream`, which exist unchanged on ARM.

### 2. Clickable address → Google Maps links
Commit **`106e31b`**. Pure regex + GUI, no backend dependency.

- **transcriber.py** (+128 lines, all standalone — very low conflict risk):
  `extract_addresses(text)`, `maps_url(query, location)`, plus helpers
  `_NOT_STREET_WORDS`, `_looks_like_street`, `_NAME_NT`, and the regexes
  `_ADDR_NUMBERED` / `_ADDR_NAMED` / `_ADDR_INTERSECTION` / `_XNAME`.
- **gui.py**: `_insert_message_text()` (splits the message body to insert
  clickable `addr:N` spans), `LINK_FG`, `_feed_location()`, `_open_map()`, the
  per-feed **Location** field in `AddStreamDialog`, and `location` presets on
  `FEED_CATALOG` entries.
- **tests**: `test_address.py` (23 checks incl. false-positive traps).
- **ARM caveat**: none. If your ARM feeds are the same Cleveland set, the
  `FEED_CATALOG` locations apply verbatim.

### 3. Scroll-position preservation
Commit **`f31eda9`**. Pure GUI.

- **gui.py**: `_at_bottom(widget)` staticmethod (`yview()[1] >= 0.999`);
  `_render_line` captures `stick = self._at_bottom(t)` **before** inserting and
  only calls `t.see("end")` when `autoscroll and stick` (both unified + sectors
  branches); the same gate on `w.see(idx)` inside `_highlight_spoken`.
- ⚠️ **This commit also bumps `installer/Transcriber.iss` and the `about_version`
  test to 1.1 — do NOT take those hunks.** Cherry-pick then `git checkout` the
  installer/version changes, or just hand-port the gui.py edits.

### 4. TTS persistent spoken-line highlight
Commit **`c9d9da0`**. Pure GUI + the existing TTS start/end callbacks.

- **gui.py**: `_highlight_spoken(text, on)` now uses two tags — `speaking`
  (`_HL_SPEAKING` = `#4a5a2a`, green, while reading) then `spoken`
  (`_HL_SPOKEN` = `#2e3f5c`, blue, persists until the line scrolls off), with
  `tag_raise("speaking")` so green wins on overlap.
- **ARM caveat**: Piper TTS runs on ARM (onnxruntime has ARM64 wheels), so this
  ports. Confirm `piper` + `onnxruntime` import in your ARM venv first.

---

## DO **NOT** PORT AS-IS

### A. Silero VAD packaging fix (`dc1c6f7`) — x64-only
This adds `datas += collect_data_files("faster_whisper")` to `Transcriber.spec`
because **x64 uses faster-whisper**, whose `assets/silero_vad_v6.onnx` must be
bundled or every transcription throws an ONNXRuntime "file doesn't exist". **ARM
uses whisper.cpp / pywhispercpp — a different engine with no faster_whisper
asset**, so the literal change doesn't apply.

**But apply the lesson**, because it will bite the ARM frozen build too:
- Any dependency that loads a data/model file **relative to its own `__file__`**
  at runtime must be in the spec's `collect_data_files`. On ARM, audit the
  whisper.cpp **GGUF model** path story and Piper's `espeak-ng-data` / `tashkeel`
  data.
- Grep your deps for `get_assets_path`, `Path(__file__).parent`, and
  `InferenceSession(` to find these.
- **Test transcription in the FROZEN `.exe`, not just dev.** This class of bug is
  invisible in `python gui.py` and only appears once packaged — that's exactly
  how it shipped undetected in x64 v1.0/v1.1.
- Note: the ARM `_transcribe` inherited `vad_filter=True` from the fork. When you
  swap in the whisper.cpp backend, make sure the VAD/segmenting is handled by
  your backend (or SpeechGate) and you're not silently relying on faster-whisper.

### B. Version + installer identity (`958735b`, and the `.iss` hunks above)
Keep your **own** ARM version string and installer name (e.g.
`Transcriber-ARM64-Setup`). Set `APP_VERSION` to reflect feature parity, but
don't copy `"1.2"` blindly — decide your own ARM versioning.

---

## Verify after porting

1. **Backend-independent tests — run these first** (should pass unchanged on ARM):
   `test_credentials.py`, `test_address.py`, `test_gui_smoke.py`, `test_tts.py`.
   Update the `about_version` assertion in `test_gui_smoke.py` to your ARM version.
2. **Then the real test: build the frozen ARM exe and exercise it live** — start a
   Broadcastify feed (via the new login dialog), let it transcribe, trigger a TTS
   read-aloud, click an address link, and scroll up while lines arrive. The
   packaging bugs only show up in the frozen build.

## Two hard-won gotchas from this week (x64)

- **Placeholder credentials silently drop feeds.** Fresh installs ship
  `credentials.json` with `YOUR_BROADCASTIFY_USERNAME` placeholders; the app used
  to try authenticating with that literal text. Feature #1 (login dialog +
  placeholder-ignoring `load_credentials`) is the fix — it's the single most
  valuable item to port for ARM end-users.
- **Frozen builds hide missing-data-file bugs.** Always run the packaged app
  end-to-end (transcribe + TTS) before shipping, not just the dev entrypoint.
