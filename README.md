# Stream Transcriber (police / emergency radio)

Live transcription of one or more audio **stream URLs** (e.g. Broadcastify scanner
feeds) using Whisper **large-v3** on your local GPU via
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). It reads the stream
directly, so **nothing plays on your speakers** — the program hears the feed while
you stay muted. Multiple feeds run at once and share the GPU.

Everything runs locally on your machine. No audio leaves your computer.

## Requirements

**Windows 10/11, 64-bit.** Both x64 and ARM64 are supported — the app picks its
transcription engine to match:

| | **x64 (recommended)** | **Windows on ARM (Snapdragon)** |
|---|---|---|
| Engine | faster-whisper (ctranslate2) | whisper.cpp (pywhispercpp) |
| Compute | NVIDIA GPU (CUDA), or CPU | CPU |
| Model | large-v3 works well on GPU | use a small one (`base.en` / `small.en`) |
| Setup | `gui.bat` | build from source — see **[BUILD_ARM.md](BUILD_ARM.md)** |

- **An NVIDIA GPU with CUDA support** is recommended on x64 — large-v3 across
  several feeds needs one. Without a GPU the app now falls back to the CPU
  automatically (pick **Device: CPU** and a smaller model in the toolbar); note
  large-v3 on CPU runs at roughly real time or slower, so choose `base.en`/`small.en`.
- **ARM64 has no CUDA** (there's no NVIDIA GPU); it runs whisper.cpp on the CPU,
  which is comfortably faster than real time with a small English model.
- **Internet on first launch** — it downloads the speech model once (and, on x64
  GPU, the CUDA runtime ~1 GB), then runs fully offline.

Text-to-speech works on both: neural **Piper** voices where available, otherwise
the built-in **Windows voices** (WinRT/OneCore, or SAPI5).

> ARM caveat: per-app ("application") audio capture is unavailable — `proc-tap`'s
> native component has no ARM64 build. URL feeds, PC-audio loopback and Stereo Mix
> all work.

## GUI (recommended)

Double-click **`gui.bat`** for the desktop app. It has:

- **Feeds** (toolbar button / Streams menu) — the single window for everything
  feed-related: a persistent **library** of every feed you've saved. Each row
  has **Add** (start transcribing) / **Remove** (stop, but keep it saved) /
  **Edit** / **Delete** (forget entirely). **+ Add new feed** saves a new feed to
  the library without auto-starting it. All changes save to `config.json`.
- **Two views** (View menu or the toolbar dropdown):
  - **Unified** — one combined feed, each line tagged with its sector/stream.
  - **Sectors** — a separate scrolling panel per stream, side by side (the
    default). **Drag a column's header** (the ⠿ handle) left or right onto
    another column to reposition it — a floating label follows your cursor and
    the drop-target column highlights. The dragged column lands where you drop
    it; the new order is saved. The chosen view is also remembered across
    restarts.
- **Listen to:** dropdown — hear **one** stream at a time through your speakers
  while every stream keeps transcribing. Set it to `(none)` to stay fully muted.

The model loads in the background on startup (status bar shows progress); the
window is usable immediately. Transcripts still save to `logs/` as well.

## Quick start (CLI)

1. Put your feed URL(s) in `config.json` (set `"disabled": false`):

   ```json
   "streams": [
     { "name": "POLICE1", "url": "https://.../your-feed", "color": "cyan", "disabled": false }
   ]
   ```

2. Double-click **`run.bat`** (or run it from a terminal).

   First run downloads the large-v3 model (~3 GB) once. After that it's instant.

3. Watch the console. Each transmission appears as:

   ```
   [14:03:21] POLICE1    Dispatch, show me en route to that call, code three.
   ```

   Transcripts are also saved to `logs/<STREAM>-<DATE>.log`.

Press **Ctrl+C** to stop.

## Getting a feed URL

You picked Broadcastify-style online feeds. The app needs the **direct audio
stream URL** (an `.mp3`/`.aac`/`.m3u8` endpoint), not the web player page.
For Broadcastify, the direct stream URL is available on the feed page / via a
Premium subscription. Any direct HTTP(S) audio stream that ffmpeg can open works.

## Transcribing PC audio (Fubo, YouTube, anything you hear)

Besides stream URLs, the app can transcribe **audio playing on your PC**. In
**Add custom stream**, set **Source = "pc audio"** and choose the **speakers
(output device)** to capture — e.g. *Speakers (Realtek)*, an external DAC, or
HDMI. It captures that output directly (via the `soundcard` library's loopback —
more reliable than the old "Stereo Mix") and transcribes it as its own sector
alongside your scanners.

PC-audio sources auto-use a **fast ~6-second flush** (continuous audio rarely has
the silence gaps that end a scanner transmission, so without this it would lag by
up to ~25s). To trade latency vs. fewer line-breaks, add a per-stream override in
`config.json`, e.g. `"vad": {"max_segment_sec": 4.0}` on that stream. (Whisper
still needs a moment to transcribe each chunk, so expect a few seconds of delay.)

**Not sure which speakers to pick?** Right-click a PC-audio sector → **"Change
audio source…"**, make sure your audio is playing, and click **🔊 Detect
signal** — it samples each output device's live level and selects the one your
sound is actually coming from. You can also right-click any sector to remove it.

Notes specific to PC audio:

- **Pick the output your audio actually uses.** If your TV plays through an
  external DAC but you capture the Realtek speakers, you'll get silence. Use
  Detect signal to be sure.
- **Muting:** capturing the output means fully muting that device silences the
  transcript too. Lower the *app's* slider in the Windows Volume Mixer instead.
- **No call signs**, so unit-coloring stays white; stream coloring still applies.
- **Remote Desktop:** while connected via RDP, local outputs are replaced by
  "Remote Audio" and won't capture the physical machine's sound — do PC-audio
  capture at the physical console.

### Capturing one specific application (per-app)

Set **Source = "application"** in Add custom stream to transcribe a single app's
audio directly (Windows process loopback). Click **↻** to list apps currently
playing audio, pick one, and it captures just that app — regardless of which
speakers it uses, and even alongside other sounds. Requires the `proc-tap`
package (installed); the option is hidden if unavailable.

Caveats:
- The app must be **playing audio** to appear in the list — start playback, then ↻.
- Captures a process **and its children**. Apps that share one process tree
  (e.g. all Chrome tabs) can't be separated from each other — use the
  output-device routing approach in `APP_ROUTING.md` for that, or different
  browsers.

The older approach — route each app to a different output device, then capture
the device — still works and is documented in `APP_ROUTING.md`; for more separate
sources than you have hardware outputs, see `MULTICHANNEL.md`.

## Read aloud (text-to-speech)

Hard to make out the radio? Have the app **read the transcription aloud** in a
clear neural voice (Piper). Click **Speak** in the toolbar:

- **Read transcriptions aloud** — the on/off master switch.
- **Voice** — pick a downloaded Piper voice.
- **What to read**:
  - *Selected feeds* — read everything from the feeds you check (default).
  - *Only lines with my keywords* — read only transmissions containing a keyword
    (e.g. a street, "shots fired", a unit) — great for monitoring quietly.
  - *Selected feeds + keyword matches* — both.
- **Feeds to read** — check one or several.
- **Alert keywords** — check preset categories (Shooting, Fire, Suspect, Crash,
  Hostage, Pursuit, Officer needs help, Weapon, Medical, Robbery/Burglary,
  Assault, Missing person, Explosion, Injury). Each preset expands to several
  synonyms automatically (e.g. Shooting = shots / gunshots / gunfire / shots
  fired / …). Add your own in the **Extra keywords** box (comma-separated).
  All matched case-insensitively.
- **🔊 Test voice** speaks a sample so you can check it.

Utterances are spoken one at a time (no overlap), and if a feed gets very busy
the queue drops the oldest so speech never lags far behind. **The line being read
is highlighted** in the transcript so you can follow along. Only **active
(transcribing) feeds** appear in the "Feeds to read" list. Settings are saved to
`config.json` and restored next launch.

Six voices are bundled (US male/female + British). Switch anytime in the Voice
dropdown — it reloads on Save.

**Voice models** live in `tts_voices/` (git-ignored; ~63 MB each). To add more:
`.venv\Scripts\python.exe -E -c "from pathlib import Path; from piper.download_voices import download_voice; download_voice('en_US-ryan-high', Path('tts_voices'))"`
then pick it in the Voice dropdown. Browse voice ids at
[the Piper voices list](https://github.com/OHF-Voice/piper1-gpl/blob/main/VOICES.md).

## Changing / updating the Whisper model

The transcription engine is **Whisper** (not a chat LLM). Pick the model from the
toolbar **Model:** dropdown — it reloads live (~30–70s) **without stopping your
feeds**, downloads the model once (cached), and saves your choice to `config.json`.

- `large-v3` — best accuracy (default).
- `large-v3-turbo` — faster, near-large accuracy.
- `distil-large-v3` — ~2× faster, ~95% accuracy; good with many feeds.
- `medium` / `small` — progressively faster/lighter, less accurate.

### Checking for updates

The app **checks PyPI for newer `faster-whisper` / `ctranslate2`** automatically
on launch (quietly — just a one-line note in the status bar if something's
available), and you can press **Check updates** anytime for a full report. It
**only reports** — it never installs anything, so it can't re-trigger the
Python-version wheel trap that broke the original setup. If you're offline the
check fails silently.

When an update is offered, apply it deliberately (in the project folder, then
restart the app):

    .venv\Scripts\python.exe -E -m pip install -U faster-whisper ctranslate2

After updating the library, any newly-supported Whisper models can be added to
`MODEL_CHOICES` in `gui.py` (or set `"model"` in `config.json`) to appear in the
dropdown. The `-E` flag avoids this machine's PYTHONPATH trap — see below.

## Tuning

`config.json`:

- `model` — `large-v3` (best), or `distil-large-v3` / `small` for more speed.
- `initial_prompt` — primes Whisper with radio vocabulary; edit to add your local
  unit names, street names, or common 10-codes for better accuracy.
- `vad.trigger_ratio` — how far above the noise floor counts as a transmission.
  Raise it if static triggers false transcripts; lower it if quiet calls are missed.
- `vad.silence_hangover_sec` — silence gap that ends a transmission.
- `filters.max_no_speech_prob` / `min_avg_logprob` — drop low-confidence/garbage
  output (helps suppress Whisper "hallucinations" on static).
- `log_retention_days` — delete log files older than this many days (default 14).
  `0` keeps them forever.
- `clips.enabled` / `clips.retention_days` / `clips.max_gb` / `clips.bitrate` —
  audio clip recording (off by default). See "Retention" and "Clip recording".

### Retention

**Streams → Retention…** sets how long saved data is kept. Transcripts and clips
are configured separately, because their sizes aren't comparable — a day of text
is kilobytes, a day of audio is tens of megabytes. One shared number would either
throw away cheap transcripts early or keep expensive audio far too long.

| | Setting | Default |
|---|---|---|
| Transcripts (text) | `log_retention_days` | 14 days |
| Audio clips | `clips.retention_days` | 7 days |
| Audio clips | `clips.max_gb` — optional size cap | off |

`0` days means "keep forever" for either.

The **size cap** is the setting that actually bounds the disk. Days alone can't:
a busy week and a quiet one differ by an order of magnitude at the same
retention. When the clips folder exceeds the cap, the **oldest clips are deleted
first** until it fits.

Age is applied before the cap, so the cap only ever has to evict clips that are
still inside their retention window. Day-based purging runs at startup; the size
cap is *also* enforced while the app runs, so a session left up for days can't
sail past it.

**Apply and purge now** in the dialog runs both policies immediately against the
values on screen, so you can see the effect before committing.

Saved `.tscript` files are never touched by any of this — deleting those is
always your call.

The GUI keeps recent transcript lines in memory and **replays them** when you
toggle a feed or switch views, so the visible scrollback no longer resets.
Use **View → Clear** to wipe it intentionally.

**Font size** — use the **A− / A+** buttons (top-right of the toolbar), or
**Ctrl + / Ctrl − / Ctrl 0** (zoom in / out / reset), to resize the transcript
text. It applies to all panels instantly and is remembered across restarts
(`font_size` in `config.json`).

## Color by unit / call sign

**The message text is white by default and takes a speaker's color whenever a
unit/call sign is identified** in that transmission — so identified speakers
stand out and everything else stays plain white. Each unit keeps the same color
all session.

The toolbar **Color by:** dropdown additionally controls the line *label*:

- **stream** — the prefix is the feed/sector name in its color (default).
- **unit** — the prefix is the **call sign** (e.g. *Adam 33*, *Engine 14*),
  and call signs are **clickable** to filter (see below).

**How it works & its limits.** Whisper transcribes speech but has no concept of
*who* is talking. True acoustic speaker-ID ("diarization") is unreliable on
compressed, short radio bursts. Instead this detects the unit each transmission
*announces* — far more accurate on police/fire radio. Tradeoff (by design):

- Lines where no one states a call sign stay plain **white** — it won't guess.
- It deliberately **ignores license plates and addresses** spelled with
  phonetics/numbers (e.g. "King X-Ray Edward 1-9-4-2") so they aren't mislabeled
  as units. Precision is favored over catching every unit.
- Add local prefixes (e.g. department-specific words) via `unit_prefixes` in
  `config.json`, e.g. `"unit_prefixes": ["zone", "sierra"]`.

### Follow one unit (click-to-filter)

With **Color by: unit** on, the call-sign prefix is **clickable** — click one to
filter the view to just that unit — handy for following one car/engine.
The filter bar (second toolbar row) shows the active unit; click **Show all
units** to clear it. Switching back to **Color by: stream** clears it too.
Filtering only affects what's shown — nothing is lost from the logs or history.

## Feeds window

The **Feeds** button (toolbar / Streams menu) is the single place to manage
everything — a persistent list of every feed you've saved (the built-in Greater
Cleveland feeds **plus anything you add**). Per row:

- **Add** — start transcribing this feed.
- **Remove** — stop transcribing it, but **keep it saved** (re-add anytime, no
  re-typing the URL). Active feeds show a green "active" label with this button.
- **Edit** — fix a typo, rename, change color, or update a changed URL. If the
  feed is currently active, it restarts live with the new settings.
- **Delete** — permanently forget the feed.
- **PDF** — save that feed's transcript as a PDF (see below).
- **+ Add new feed** — add a brand-new feed (URL or feed id). It's saved to the
  library but does **not** start transcribing until you click its **Add**.
- **Reorder** — drag the **⠿** handle on the left of a row up or down to change
  the feed order. The new order is saved and restored next launch.

The built-in Cleveland feeds are: Cleveland West, Cleveland Citywide (covers east
side), Cleveland Fire/EMS, Westlake/WestCom, and East Cleveland.

### Feed service — police, fire/EMS, ATC, general

Each feed can say what kind of radio it carries, in **Feeds → Edit → Service**.
It isn't cosmetic: three parts of the pipeline are domain-specific, and a service
sets all three together.

| Service | Whisper prompt | Call signs | Links |
|---|---|---|---|
| *(blank)* | the global `initial_prompt` | police + fire | addresses |
| **Police** | police wording | police + fire | addresses |
| **Fire / EMS** | fire/EMS wording | police + fire | addresses |
| **Air traffic control** | aviation phraseology | airline + tail numbers | **aircraft → FlightRadar24** |
| **General** | none | off | none |

**Police and Fire/EMS differ only in the prompt.** The call-sign extractor already
handles both at once, and shared PD+Fire dispatch channels are common — three of
the five stock Cleveland feeds are joint — so splitting it would only lose units
on the feeds that need them most. Pick whichever wording fits the traffic better.

**Air traffic control is the one that changes behaviour**, and it matters:

- The police prompt actively damages ATC audio — it turned *"hold short of
  runway"* into *"whole sort of runway"* and invented text that was never said.
- Aircraft identify as `Delta 510` / `Speedbird 117 heavy` / `November 65 Juliet
  Charlie`, which the police extractor can't see (it caught `Delta` only because
  "delta" is in its NATO prefix list, and missed United and Speedbird entirely).
- Address detection is switched **off**, because `November 65 Juliet Charlie` was
  otherwise detected as the street address *"65 Juliet Charlie"*.

**Prompt override.** The box under Service tunes one feed without touching the
rest: empty uses the service preset, and a feed with no service falls back to the
global `initial_prompt`. Resolution is *override → preset → global*, so existing
feeds behave exactly as they always did until you choose otherwise.

Both `service` and any override travel with an exported feed list.

#### Aircraft → FlightRadar24

On an ATC feed, recognised aircraft become green links: flights open
`flightradar24.com/data/flights/dl510`, registrations open
`…/data/aircraft/n65jc`. This only builds a URL for your browser — no API, no
scraping. A link can miss if FR24 isn't tracking that flight right now, the same
way a map link can miss on a garbled street name.

Labels are canonicalised before use: Whisper hyphenates spoken digits
unpredictably (`Delta 5-10` on one line, `Delta 510` the next) and controllers
append a weight class, so `DELTA 510` is derived from the parts. Without that,
one aircraft would scatter across several "units" and click-to-filter wouldn't
group it.

### Exporting / importing the feed list

**Export list…** (Feeds window, or Streams → Export feed list…) writes your whole
library to a small JSON file. **Import list…** reads one back and merges it in.

- Imported feeds are **saved only** — nothing starts transcribing until you click
  **Add** on the row.
- On a name collision you pick once for the whole import: **skip** (keep yours),
  **replace** (take theirs), or **keep both** (imports as `Name (2)`).
- Import also accepts a raw `config.json` from another install, so you don't have
  to export first if you already have one in hand.

**x64 ↔ ARM64:** the file carries feeds only — never `model` / `device` /
`compute_type` / `engine` — so a list moves cleanly between the two builds. That
matters because the two installs keep **separate** data directories
(`%APPDATA%\Transcriber` vs `%APPDATA%\Transcriber-ARM64`) and have very
different engine defaults (large-v3 on CUDA vs. a small whisper.cpp model);
carrying engine settings across would hand an ARM machine a model it can't run.

URL feeds are fully portable. Two feed types are bound to the machine that
created them and are reported as warnings after import rather than silently
dropped:

| Feed type | Travels? | Why |
|---|---|---|
| URL / Broadcastify | Yes | Just a URL |
| `pc audio` | Needs re-pointing | Names a specific output device on the old PC |
| `application` | No on ARM64 | Per-app capture has no ARM64 build; the pid is session-only |

### Clip recording — click a line, hear it again

Transcriber can keep the audio behind every transcript line. Lines with a saved
recording get a violet **🔊**; click it to hear that exact transmission.

No continuous recording is involved. The voice gate already splits each feed into
one segment per transmission, and that same segment is what gets transcribed — so
a line and its audio are 1:1 by construction, including the `preroll_sec` lead-in
so clips don't sound chopped.

Two switches, both **off** by default:

1. **Streams → Audio recording…** — the master switch, retention in days, what's
   on disk now, open-folder, and delete-all.
2. **Feeds → Edit → "Save audio for each line"** — per-feed opt-in.

Playback takes over the speakers briefly; the live feed you were listening to
resumes when the clip ends, and clicking another 🔊 interrupts the first.

| | Size |
|---|---|
| Raw 16 kHz mono WAV | ~32 KB/s → **~800 MB/day** for a busy feed |
| Opus (what's used) | ~7× smaller → **~100 MB/day** |

Encoding goes through the ffmpeg that already decodes the streams, so there's no
new dependency on either build. If that ffmpeg has no `libopus`, clips fall back
to WAV and the status line says so once — shorten the retention if that happens.

Clips are voice recordings of live radio. They stay on your PC, are purged on the
same kind of schedule as the logs (`clips.retention_days`), and can be wiped from
the recording dialog at any time.

#### Exporting clips as an MP3

Click a line to select that whole row, drag up or down to take more, then
**right-click → Export selected audio as MP3…** (or Streams → the same). Every
selected line that has a 🔊 is decoded, joined in transcript order with a short
silence between transmissions, and written as a single MP3. Selecting one line
exports just that one.

Selection is by whole rows, never part of one — a transmission either goes in the
MP3 or it doesn't. Shift-click extends to the clicked row, dragging past the top
or bottom edge scrolls, and the right-click menu has **Select all** plus a count
of how many clips your selection covers before you commit.

Joining happens as raw PCM, not by concatenating the Opus files — stitching
compressed frames would need matching encoder state. Clips decode, join, then
re-encode once at 64 kbps mono, which is ample for 16 kHz speech. Lines without
audio are skipped silently; clips already purged are reported in the status bar
rather than failing the export.

Encoding runs on a worker thread, so a long selection never freezes the window.

#### Transcript files (`.tscript`) — text and audio in one file

A transcript file keeps the lines **and the audio behind them** together, so an
incident can be archived or handed to someone else. Unlike the live logs and
clips, it isn't touched by any retention setting.

- **Save selected as transcript file…** — right-click a row selection.
- **Streams → Save whole day as transcript file…** — pick a feed and day.
- **Streams → Open transcript file…** — opens it for review, exactly like a past
  day: the text, with 🔊 on every line whose audio came along.

It's an ordinary **ZIP with a different extension** — deliberately, not a private
format. Rename it `.zip` and anything can read it:

```
manifest.json     format/version, feed, day, span, counts
transcript.json   [{"ts", "text", "clip"}, ...]
clips/*.opus      only the clips those lines reference
```

That matters for something meant to be an archive: the text is JSON and the audio
is ordinary Opus, so it stays readable even without Transcriber. `zipfile` is
stdlib, so this adds no dependency on either build.

Clips are decoded **from inside the archive** rather than extracted, so opening
one doesn't scatter copies of voice recordings through temp folders. Lines whose
clip was already purged are still saved — they just carry no audio, and the
status bar says how many.

> **These files deliberately outlive `clips.retention_days`.** That's the point
> of having them, but it does mean `purge_old_clips` can't reach inside one. They
> contain PII and recorded voices: keep them somewhere you'd be comfortable
> keeping the logs, and delete them yourself when done.

#### Playing a transcript through

Any transcript you're reviewing — an opened `.tscript` or a past day — gets a
transport bar with **two** ways to play it:

| | What it plays | Works on |
|---|---|---|
| **▶ Play audio** | the recorded radio audio, clip after clip | lines that still have a 🔊 |
| **🗣 Read aloud** | the voice reading the transcribed text | every line |

Both step one line at a time, highlighting the current line and scrolling to keep
it in view. **⏸ Pause** holds your place, **⏹ Stop** rewinds to the top.

The distinction matters more than it first looks: **Play audio** skips lines whose
clips have been purged, while **Read aloud** works on any transcript at all —
including an old day whose audio is long gone.

Select a line before pressing play to start from there rather than the beginning.

#### Reviewing an earlier day

The live window restores **today's** lines on launch. To go further back, use
**Streams → Open a past day…**: pick a feed, then a day. The list is newest-first
and shows what each day holds:

```
2026-07-28   103 lines  🔊 103
2026-07-19   182 lines
2026-07-15   285 lines
```

The 🔊 count is audio still on disk. Days without it kept their transcript but
their clips have been purged — `clips.retention_days` (7) is shorter than
`log_retention_days` (14), so the older stretch is text-only by design.

Opening a day takes over the window with a read-only transcript of it, marked
with 🔊 wherever the audio survived, and an amber banner naming the day. Feeds
keep transcribing and recording the whole time — new lines are held in memory
and appear when you click **Back to live**.

```jsonc
"clips": {
  "enabled": true,        // master switch
  "retention_days": 7,    // 0 = keep forever
  "bitrate": "24k"        // Opus target
}
```

### Saving a transcript as PDF

Streams → **Save transcript as PDF…**, or the **PDF** button on a feed row. Choose
the feed, then what to include:

- **This session** — what's currently on screen for that feed.
- **Saved logs** — one day, several (Ctrl/Shift-click), or all of them if you
  select none. Only days still on disk appear (see `log_retention_days`).

Page 1 opens with the feed name in large bold type, and under it the span of the
export — the date and time of the **first and last transmission it contains**,
not the day you picked:

```
Westlake/WestCom
2026-07-28  ·  10:38:24 – 11:05:02  ·  116 transmissions
────────────────────────────────────────────────────────
[10:38:24] Medics 7-2 in route 10-37.
```

An export spanning several days dates both ends
(`2026-07-12 08:00:01 – 2026-07-19 23:59:12`). Log files only record the time of
day, so the date comes from the filename.

Output is a timestamped, wrapped, page-numbered PDF. The writer is plain stdlib —
no reportlab — so it behaves identically on the x64 and ARM64 builds and adds
nothing to either installer.

## How it works

`ffmpeg` opens each stream URL and emits 16 kHz mono PCM → an adaptive
energy/voice **gate** carves it into individual transmissions (tracking the
background noise floor so squelch/static doesn't trigger it) → a single shared
**Whisper large-v3** GPU worker transcribes each transmission → results print to
the console and append to per-stream logs.

## Important environment note

This project runs on a **Python 3.13 venv** (`.venv`). Your machine has a global
`PYTHONPATH` pointing at the 3.14 site-packages, which corrupts any venv unless
ignored. `run.bat` launches Python with **`-E`** (isolated mode) to ignore it.
Always launch via `run.bat`, or with `.venv\Scripts\python.exe -E transcriber.py`.

(Consider removing that global `PYTHONPATH` environment variable — it will cause
similar breakage for any other Python project on this machine.)

## Legal

Listening to and transcribing **unencrypted** public-safety radio for personal
use is legal in the US (Ohio included for home use). Don't attempt to decrypt
encrypted channels, and check your feed provider's Terms of Service before
**redistributing** transcripts — personal use is fine.
