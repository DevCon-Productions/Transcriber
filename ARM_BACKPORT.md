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
| `41fb934` | GitHub-release app self-updater | ✅ Yes — **but one critical ARM change** (see #5) |
| `99c16f3` | v1.3 version bump | ❌ Keep your own ARM version |

> Note: the repo is now **public** and its history was scrubbed (a username was
> removed from an old test commit), which is why some hashes above differ from
> what you may have seen earlier. `git fetch origin master` to get the current
> ones. The public repo is what makes the self-updater's anonymous downloads work.

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

### 5. GitHub-release self-updater — port it, but with ONE critical ARM change
Commits **`41fb934`** (feature) + **`99c16f3`** (v1.3 bump). Checks the public
repo's latest release and, if newer, downloads the installer and upgrades in
place. Mostly backend-agnostic, but multi-architecture makes the **asset picker**
a landmine.

- **transcriber.py**: `APP_REPO`, `check_for_app_update(current_version)` (hits
  GitHub `/releases/latest` anonymously → `is_newer(tag, APP_VERSION)` → returns
  the installer asset + notes), `download_file(url, dest, progress_cb)` (streams
  to a `.part`, renames on success).
- **gui.py**: `AppUpdateDialog` (notes → Download & install → `ttk.Progressbar`),
  `_check_app_update_manual/_auto`, `_run_app_update_check`,
  `_handle_app_update_result`, `_start_app_download`, `_download_update`,
  `_launch_installer_and_quit`, `_open_url`; the **Help → "Check for app updates"**
  menu item; the four event kinds (`app_update_result` / `app_dl_progress` /
  `app_dl_done` / `app_dl_error`) in `_drain_events`; `self._update_dialog` init;
  and the `self.root.after(3500, self._check_app_update_auto)` quiet check.
- **tests**: `test_appupdate.py` (14 checks, network mocked).

⚠️ **CRITICAL — the asset picker must be architecture-aware.** In transcriber.py,
`check_for_app_update` currently grabs the **first `.exe`** asset in the release:
```python
for a in data.get("assets", []):
    if str(a.get("name", "")).lower().endswith(".exe"):
        asset = a; break
```
On ARM this could download the **x64 installer**. Before porting, decide the
release strategy and fix the picker:

- **Recommended — one release, two assets.** Attach both
  `Transcriber-Setup-<v>.exe` (x64) and `Transcriber-ARM64-Setup-<v>.exe` (arm64)
  to each GitHub release. Then the ARM build selects the asset whose name contains
  `arm64`; the x64 build selects the one that does **not**. (Tell the x64 session
  to make the same tweak, or x64 could grab the ARM installer once ARM assets
  exist — flagged in the x64 memory too.)
- **Alternative — separate release channels/tags for ARM.** Then point the check
  at the ARM releases (or filter tag names) so `/releases/latest` never hands the
  ARM app an x64 build.

**Other ARM notes:**
- **Versioning:** keep ARM version numbers parallel to x64 (both `"1.4"`, etc.) so
  `is_newer(tag, APP_VERSION)` just works. If you use an ARM suffix, adjust the
  tag parse in `check_for_app_update`.
- **Installer:** your ARM Inno `.iss` needs its **own fixed `AppId`** (distinct
  from x64's `{{8F3C1A22-…-TRANSCRIBER01}}`) so the two arches are separate
  products, plus the admin manifest and the `[Run]` postinstall relaunch. With
  those, `os.startfile(installer)` → UAC → in-place upgrade → relaunch all behave
  the same on ARM Windows. No code change needed in `_launch_installer_and_quit`.
- **Bootstrap caveat (same as x64):** the updater only activates from the ARM
  version it first ships in. The first ARM release carrying the updater must be
  installed manually; ARM releases after that self-update.
- Everything else — the dialog, progress download, event wiring, launch-and-quit
  — is backend-agnostic and ports unchanged.

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
