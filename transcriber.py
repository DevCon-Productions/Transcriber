"""
Police-radio (and general audio-stream) live transcriber.

Listens to one or more audio stream URLs, detects individual radio transmissions
with an adaptive voice/energy gate, and transcribes each one on the GPU using
faster-whisper (Whisper large-v3 on your RTX 3090). Output goes to the console
(color-coded per stream) and to per-stream log files under ./logs.

Because it reads the stream URL directly, nothing is ever played on your
speakers -- the program "hears" the feed while you stay on mute. Multiple
streams run concurrently and share the single GPU through one worker queue.

Run it with the isolated-mode launcher (run.bat) or:
    .venv\\Scripts\\python.exe -E transcriber.py

NOTE: -E is required on this machine. A global PYTHONPATH points at the 3.14
site-packages and will corrupt this 3.13 venv if not ignored. run.bat handles it.
"""

import os
import re
import sys
import glob
import json
import math
import time
import queue
import base64
import platform
import threading
import subprocess
import collections
import datetime as dt


# --------------------------------------------------------------------------
# Interpreter architecture.
#
# On Windows-on-ARM, platform.machine() is NOT a reliable signal for the running
# interpreter: an EMULATED x64 Python reports 'ARM64' too (machine() reflects the
# host, not the process). The real process architecture is in
# PROCESSOR_ARCHITECTURE -- 'AMD64' for an emulated/native x64 process, 'ARM64'
# only for a native ARM64 one. We key CUDA and backend selection off THIS, so an
# x64 build running emulated on an ARM box still (correctly) uses the CUDA/
# ctranslate2 path that its amd64 wheels support.
# --------------------------------------------------------------------------
def interpreter_is_arm64():
    """True only if this Python process is a native ARM64 build (not emulated x64)."""
    proc = os.environ.get("PROCESSOR_ARCHITECTURE", "").upper()
    if proc in ("AMD64", "X86", "X64", "IA64"):
        return False
    if proc in ("ARM64", "AARCH64"):
        return True
    # Fallback (non-Windows / unusual): trust platform.machine().
    return platform.machine().lower() in ("arm64", "aarch64")


# --------------------------------------------------------------------------
# Environment bootstrap: make the pip-installed NVIDIA CUDA DLLs discoverable
# by ctranslate2, and locate the bundled ffmpeg binary.
# --------------------------------------------------------------------------
def _nvidia_search_roots():
    """All places CUDA DLLs might live: the dev venv site-packages, a frozen
    bundle's _internal/nvidia, and the per-user CUDA dir the slim installer
    downloads into on first run."""
    roots = []
    # Dev venv: .../Lib/site-packages/nvidia
    roots.append(os.path.abspath(os.path.join(
        os.path.dirname(sys.executable), "..", "Lib", "site-packages", "nvidia")))
    # Frozen bundle (if CUDA was bundled): next to the exe
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        roots.append(os.path.join(base, "nvidia"))
    # Slim-installer per-user download location
    roots.append(os.path.join(_user_data_dir(), "cuda", "nvidia"))
    return roots


def add_nvidia_dll_dirs():
    if interpreter_is_arm64():
        return                      # no CUDA on ARM; whisper.cpp uses CPU/NPU
    for site in _nvidia_search_roots():
        for bindir in glob.glob(os.path.join(site, "*", "bin")):
            try:
                os.add_dll_directory(bindir)
            except (FileNotFoundError, OSError):
                pass
            os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


def cuda_libraries_present():
    """True if the core CUDA runtime DLLs (cublas + cudnn) are findable on the
    search paths — i.e. the model can run on GPU without downloading anything."""
    needed = ("cublas64", "cudnn64")
    for site in _nvidia_search_roots():
        found = {n for n in needed
                 for _ in glob.glob(os.path.join(site, "*", "bin", n + "*.dll"))}
        if len(found) == len(needed):
            return True
    return False


# Package versions pinned to what this app was built/tested against.
CUDA_PACKAGES = [
    "nvidia-cublas-cu12==12.9.2.10",
    "nvidia-cudnn-cu12==9.23.0.39",
    "nvidia-cuda-nvrtc-cu12==12.9.86",
]


def ensure_cuda_libraries(status_cb=None):
    """Slim-installer first run: if the CUDA runtime DLLs aren't present, fetch
    them into the per-user data dir and put them on the DLL search path. Returns
    (ok: bool, message: str). Needs internet the first time only. `status_cb(str)`
    receives progress lines. No-op (returns True) if CUDA is already available."""
    def say(m):
        if status_cb:
            try:
                status_cb(m)
            except Exception:
                pass

    if interpreter_is_arm64():
        # No CUDA on Windows-on-ARM; the whisper.cpp backend runs on CPU/NPU.
        return True, "CUDA not applicable on ARM (using CPU/NPU backend)."
    if cuda_libraries_present():
        return True, "CUDA libraries present."

    target = os.path.join(_user_data_dir(), "cuda")
    os.makedirs(target, exist_ok=True)
    say("Downloading GPU libraries (one-time, ~1 GB)…")

    # Install the pinned nvidia wheels into `target` using pip. Prefer a real
    # python interpreter; in a frozen build fall back to pip's in-process API.
    try:
        py = _find_python_for_pip()
        if py:
            import subprocess
            cmd = [py, "-m", "pip", "install", "--no-cache-dir",
                   "--target", target] + CUDA_PACKAGES
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  **_no_window_kwargs())
            if proc.returncode != 0:
                return False, f"GPU library install failed:\n{proc.stderr[-800:]}"
        else:
            from pip._internal.cli.main import main as pip_main
            rc = pip_main(["install", "--no-cache-dir", "--target", target]
                          + CUDA_PACKAGES)
            if rc != 0:
                return False, "GPU library install failed (pip returned non-zero)."
    except Exception as e:
        return False, f"Could not install GPU libraries: {e}"

    add_nvidia_dll_dirs()   # pick up the freshly-installed DLLs
    if cuda_libraries_present():
        say("GPU libraries ready.")
        return True, "GPU libraries installed."
    return False, "GPU libraries installed but still not found on the search path."


def _find_python_for_pip():
    """A python.exe we can call with -m pip. In dev that's sys.executable; in a
    frozen build sys.executable is the app, so look for a system python."""
    if not getattr(sys, "frozen", False):
        return sys.executable
    import shutil
    for name in ("python.exe", "python3.exe", "py.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def find_ffmpeg():
    """Path to an ffmpeg binary, in preference order:
    1. the one bundled by the imageio-ffmpeg pip package (x64 only -- that package
       has no ARM64 wheel, so this simply misses on ARM),
    2. an ffmpeg shipped with the app (bin/ next to it, or the user data dir) --
       this is the ARM path; see BUILD_ARM.md for staging a native ARM64 build,
    3. whatever is on PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for d in (os.path.join(HERE, "bin"), HERE, os.path.join(DATA_DIR, "bin")):
        p = os.path.join(d, exe)
        if os.path.isfile(p):
            return p
    return "ffmpeg"  # fall back to a system ffmpeg on PATH


def ffmpeg_available(ffmpeg=None):
    """True if the resolved ffmpeg can actually be executed. URL/stream feeds need
    it -- without it every StreamWorker just loops on 'ffmpeg launch failed', so the
    Engine warns once up front instead."""
    try:
        proc = subprocess.run([ffmpeg or find_ffmpeg(), "-version"],
                              capture_output=True, timeout=10, **_no_window_kwargs())
        return proc.returncode == 0
    except Exception:
        return False


import numpy as np
# NOTE: faster_whisper / WhisperModel is imported LAZILY inside Engine.load_model
# (after CUDA DLL dirs are set up, and after the slim-installer CUDA download).
# Importing it eagerly here would force CUDA resolution at module load.


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def _resource_dir():
    """Directory of bundled READ-ONLY resources. When frozen by PyInstaller this
    is the app install dir (sys._MEIPASS / exe dir); in dev it's this file's dir."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _user_data_dir():
    """Per-user WRITABLE dir for config/credentials/logs. In a frozen install the
    app lives in Program Files (read-only), so writable state goes to AppData.
    In dev, everything stays in the project folder for convenience.

    The ARM64 build uses a DISTINCT dir (Transcriber-ARM64) so it never inherits an
    x64 install's config/credentials on the same machine -- the x64 default is
    large-v3/CUDA, which on ARM means a 3 GB download and a model that won't load.
    Lets both architectures be installed side by side with independent state."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        name = "Transcriber-ARM64" if interpreter_is_arm64() else "Transcriber"
        d = os.path.join(base, name)
        os.makedirs(d, exist_ok=True)
        return d
    return os.path.dirname(os.path.abspath(__file__))


HERE = _resource_dir()                    # read-only resources (icons, voices)
DATA_DIR = _user_data_dir()               # writable state
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")
CREDENTIALS_PATH = os.path.join(DATA_DIR, "credentials.json")
LOG_DIR = os.path.join(DATA_DIR, "logs")
# Voices may be bundled (read-only, in HERE) or user-downloaded (writable, in
# DATA_DIR). Prefer a user dir with voices; else fall back to the bundled one.
_user_voices = os.path.join(DATA_DIR, "tts_voices")
_bundled_voices = os.path.join(HERE, "tts_voices")
TTS_VOICE_DIR = _user_voices if os.path.isdir(_user_voices) else _bundled_voices

SAMPLE_RATE = 16000          # Whisper wants 16 kHz mono
FRAME_MS = 30                # VAD frame size
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * 2   # s16le -> 2 bytes/sample

ANSI_COLORS = {
    "red": "91", "green": "92", "yellow": "93", "blue": "94",
    "magenta": "95", "cyan": "96", "white": "97", "grey": "90",
}


def _no_window_kwargs():
    """Popen kwargs that prevent a console window from flashing on Windows each
    time ffmpeg is spawned (notably on every stream reconnect). No-op elsewhere."""
    if os.name != "nt":
        return {}
    flags = 0
    flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW   # hide even if a window is made
    si.wShowWindow = 0                              # SW_HIDE
    return {"creationflags": flags, "startupinfo": si}


def _seed_from_example(target, example_name):
    """On first run (installed build), copy a bundled *.example.json to the
    writable data dir so the app has a starting config/credentials file."""
    if os.path.exists(target):
        return
    src = os.path.join(HERE, example_name)
    if os.path.exists(src):
        try:
            import shutil
            shutil.copyfile(src, target)
        except Exception:
            pass


def load_config(path=CONFIG_PATH):
    # First run of an installed build: seed config + credentials from examples.
    # On native ARM64 prefer config.example.arm.json (small CPU model default) if
    # the build bundled it -- the shared config.example.json defaults to large-v3
    # on CUDA, which on ARM would mean a ~3 GB download and unusable CPU speed.
    example = "config.example.json"
    if interpreter_is_arm64() and os.path.exists(
            os.path.join(HERE, "config.example.arm.json")):
        example = "config.example.arm.json"
    _seed_from_example(CONFIG_PATH, example)
    _seed_from_example(CREDENTIALS_PATH, "credentials.example.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Characters illegal in Windows filenames -> stream names can contain any of them.
_FNAME_BAD = re.compile(r'[<>:"/\\|?*]')


def safe_filename(name):
    """Make a stream name safe to embed in a log filename (e.g. 'Fire/EMS')."""
    return _FNAME_BAD.sub("-", name).strip() or "stream"


def is_enabled(stream):
    """A stream is active unless explicitly disabled. Single source of truth.
    URL feeds need a url; pcaudio sources need either an output_device name
    (soundcard loopback) or a device index (Stereo Mix fallback)."""
    if stream.get("disabled", False):
        return False
    if stream.get("type") == "app":
        return stream.get("pid") is not None
    if stream.get("type") == "pcaudio":
        return stream.get("output_device") is not None or stream.get("device") is not None
    return bool(stream.get("url"))


# --------------------------------------------------------------------------
# Feed-list export / import.
#
# The exported file holds ONLY feed entries -- never the engine settings around
# them (model/device/compute_type/engine). That is deliberate: the x64 build
# defaults to large-v3 on CUDA and the ARM64 build to a small whisper.cpp model,
# and the two installs keep SEPARATE data dirs (see _user_data_dir). Carrying
# engine keys across would hand an ARM machine a 3 GB CUDA model it can't run.
# Feed entries themselves are plain data, so a list exported on either
# architecture imports cleanly into the other.
#
# Two fields do NOT travel meaningfully and are dropped on export:
#   pid        -- a process id; meaningless in another session, let alone another box
#   disabled   -- "currently transcribing" is per-install state, not part of the feed
# Two feed TYPES are machine-bound even though they survive the round trip:
#   pcaudio    -- names a local output device that may not exist on the target
#   app        -- per-app capture has no ARM64 build (proctap_available() is False)
# import_feeds() reports both in `warnings` rather than silently dropping them.
# --------------------------------------------------------------------------
FEED_EXPORT_FORMAT = "transcriber-feeds"
FEED_EXPORT_VERSION = 1

# Per-feed keys that are portable between machines and architectures.
FEED_PORTABLE_KEYS = ("name", "url", "type", "provider", "color", "location",
                      "desc", "output_device", "app_name", "record",
                      # What kind of radio this is, and any prompt tuning for it.
                      # Both are preferences, not machine state, so they travel.
                      "service", "initial_prompt",
                      # Which display column the feed shares (e.g. "CLE ATC").
                      "group")


def _clean_feed(entry):
    """Strip a feed entry down to its portable keys (see module notes above)."""
    return {k: entry[k] for k in FEED_PORTABLE_KEYS
            if k in entry and entry[k] not in (None, "")}


def export_feeds(path, feeds, app_version=None):
    """Write `feeds` (a list of feed dicts) to `path` as a portable JSON file.
    Returns the number of feeds written."""
    clean = [_clean_feed(e) for e in feeds if e.get("name")]
    doc = {
        "format": FEED_EXPORT_FORMAT,
        "version": FEED_EXPORT_VERSION,
        "exported": dt.datetime.now().isoformat(timespec="seconds"),
        "app_version": app_version or "",
        "feeds": clean,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    return len(clean)


def import_feeds(path):
    """Read a feed list from `path`. Returns (feeds, warnings).

    Accepts, in order of preference:
      - a file written by export_feeds()               {"format": ..., "feeds": [...]}
      - a bare JSON array of feed dicts                [{...}, {...}]
      - a whole config.json                            (uses feed_library + streams)
    The last case means a user can point this at the other architecture's
    config.json directly and still get their feeds. Raises ValueError if the file
    parses but holds no recognisable feeds."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    if isinstance(doc, list):
        raw = doc
    elif isinstance(doc, dict) and isinstance(doc.get("feeds"), list):
        raw = doc["feeds"]
    elif isinstance(doc, dict) and ("feed_library" in doc or "streams" in doc):
        # A full config.json: library first, then any active streams not in it.
        raw, seen = [], set()
        for e in list(doc.get("feed_library") or []) + list(doc.get("streams") or []):
            if isinstance(e, dict) and e.get("name") and e["name"] not in seen:
                seen.add(e["name"])
                raw.append(e)
    else:
        raise ValueError("Not a Transcriber feed list (no 'feeds' array found).")

    feeds, warnings = [], []
    for e in raw:
        if not isinstance(e, dict) or not e.get("name"):
            continue
        entry = _clean_feed(e)
        kind = entry.get("type")
        if kind == "app":
            # Per-app capture: the pid is gone (not exported) and on ARM64 the
            # proctap native module doesn't exist at all.
            if not proctap_available():
                warnings.append(f"“{entry['name']}” captures an application — "
                                "per-app capture isn't available on this build.")
            else:
                warnings.append(f"“{entry['name']}” captures an application — "
                                "re-pick the running app before starting it.")
        elif kind == "pcaudio":
            dev = entry.get("output_device")
            names = [n for n, _d in list_output_devices()]
            if dev and names and dev not in names:
                warnings.append(f"“{entry['name']}” captures speakers named "
                                f"“{dev}”, which this machine doesn't have.")
        feeds.append(entry)

    if not feeds:
        raise ValueError("No feeds found in that file.")
    return feeds, warnings


# --------------------------------------------------------------------------
# Transcript -> PDF.
#
# Written by hand against the PDF 1.4 spec rather than pulling in reportlab:
# the ARM64 build installs from a hand-curated wheel list (requirements-arm.txt)
# and every added dependency is a wheel that might not exist for win_arm64. This
# needs no dependency at all, so PDF export behaves identically on both builds.
#
# Scope matches the content: transcripts are plain text, so this emits the base-14
# fonts (Courier body / Helvetica headings), which every reader has built in and
# which need no font embedding.
# --------------------------------------------------------------------------
PDF_PAGE_W, PDF_PAGE_H = 612.0, 792.0        # US Letter, in points
PDF_MARGIN = 54.0                            # 0.75"
PDF_BODY_SIZE = 9.0
PDF_LEADING = 11.5
# Header block on page 1: feed name, then the span/line count, then a rule.
PDF_TITLE_SIZE = 17.0
PDF_SUBTITLE_SIZE = 9.5
PDF_SUBTITLE_GAP = 17.0       # title baseline -> subtitle baseline
PDF_RULE_GAP = 11.0           # subtitle baseline -> rule
PDF_HEADER_H = 58.0           # total height reserved before the body starts
# Courier is metrically fixed: every glyph is exactly 0.6 em wide.
PDF_COURIER_WIDTH = 0.6


def _pdf_escape(text):
    """Encode a string for a PDF literal. PDF's base-14 fonts use WinAnsi, so
    anything outside latin-1 (a stray smart quote from the transcriber) becomes
    '?' rather than corrupting the stream."""
    out = []
    for ch in text:
        if ch in "()\\":
            out.append("\\" + ch)
        elif " " <= ch <= "~":
            out.append(ch)
        else:
            try:
                b = ch.encode("cp1252")
            except (UnicodeEncodeError, LookupError):
                out.append("?")
                continue
            out.append("".join(f"\\{c:03o}" for c in b))
    return "".join(out)


def _pdf_wrap(text, max_chars):
    """Word-wrap to `max_chars`, breaking any single word longer than a line.
    Always returns at least one (possibly empty) line."""
    if max_chars < 8:
        max_chars = 8
    lines, cur = [], ""
    for word in text.split():
        while len(word) > max_chars:            # unbreakable run (URL, long id)
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= max_chars:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def write_transcript_pdf(path, title, lines, subtitle=""):
    """Render a transcript to a PDF at `path`.

    title    -- feed name, set large + bold as the heading on page 1
    lines    -- iterable of transcript strings, e.g. "[20:50:01] Copy that."
    subtitle -- line under the title; callers pass the transmission span and
                count (see transcript_span / format_transcript_span)

    Returns the number of pages written."""
    usable_w = PDF_PAGE_W - 2 * PDF_MARGIN
    max_chars = int(usable_w / (PDF_BODY_SIZE * PDF_COURIER_WIDTH))

    # Lay out: wrap every line, then fill pages. Continuation lines are indented
    # to the width of a "[HH:MM:SS] " stamp so the timestamp column stays readable.
    body = []
    for raw in lines:
        raw = raw.rstrip("\n")
        wrapped = _pdf_wrap(raw, max_chars) if raw.strip() else [""]
        body.append(wrapped[0])
        body.extend(" " * 11 + w for w in wrapped[1:])

    # Page 1 starts below the header block; later pages start at the margin.
    first_top = PDF_PAGE_H - PDF_MARGIN - PDF_HEADER_H
    rest_top = PDF_PAGE_H - PDF_MARGIN
    bottom = PDF_MARGIN + 16                      # room for the page footer
    first_rows = max(1, int((first_top - bottom) / PDF_LEADING))
    rest_rows = max(1, int((rest_top - bottom) / PDF_LEADING))

    pages, i = [], 0
    if not body:
        body = ["(no transcript lines)"]
    while i < len(body):
        rows = first_rows if not pages else rest_rows
        pages.append(body[i:i + rows])
        i += rows
    total = len(pages)

    def content_stream(idx, rows):
        parts = []
        y = rest_top
        if idx == 0:
            # Header block: feed name large + bold, then the span/count beneath
            # it, then a rule. Sizes and offsets track PDF_TITLE_* so the body's
            # first_top stays in step with whatever the header actually occupies.
            y = PDF_PAGE_H - PDF_MARGIN - PDF_TITLE_SIZE
            parts.append(f"BT /F2 {PDF_TITLE_SIZE} Tf 0 0 0 rg {PDF_MARGIN} "
                         f"{y:.1f} Td ({_pdf_escape(title)}) Tj ET")
            if subtitle:
                y -= PDF_SUBTITLE_GAP
                parts.append(f"BT /F2 {PDF_SUBTITLE_SIZE} Tf 0.30 0.30 0.30 rg "
                             f"{PDF_MARGIN} {y:.1f} Td "
                             f"({_pdf_escape(subtitle)}) Tj ET")
            y -= PDF_RULE_GAP
            parts.append(f"0.75 0.75 0.75 RG 0.7 w {PDF_MARGIN} {y:.1f} m "
                         f"{PDF_PAGE_W - PDF_MARGIN} {y:.1f} l S")
            y = first_top
        parts.append("0 0 0 rg")
        parts.append(f"BT /F1 {PDF_BODY_SIZE} Tf {PDF_LEADING} TL "
                     f"{PDF_MARGIN} {y:.1f} Td")
        for row in rows:
            parts.append(f"({_pdf_escape(row)}) Tj T*")
        parts.append("ET")
        foot = f"Page {idx + 1} of {total}"
        parts.append(f"BT /F1 8 Tf 0.45 0.45 0.45 rg "
                     f"{PDF_PAGE_W - PDF_MARGIN - len(foot) * 8 * PDF_COURIER_WIDTH:.1f} "
                     f"{PDF_MARGIN - 2:.1f} Td ({_pdf_escape(foot)}) Tj ET")
        return "\n".join(parts).encode("latin-1", "replace")

    # --- assemble the file: 1 catalog + 1 pages node + 2 fonts + 2 objs/page ---
    objs = {}                                    # number -> bytes (object body)
    n_catalog, n_pages, n_f1, n_f2 = 1, 2, 3, 4
    first_page_obj = 5
    page_nums = [first_page_obj + 2 * i for i in range(total)]

    objs[n_catalog] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{n} 0 R" for n in page_nums)
    objs[n_pages] = (f"<< /Type /Pages /Count {total} /Kids [{kids}] >>"
                     ).encode("latin-1")
    objs[n_f1] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier "
                  b"/Encoding /WinAnsiEncoding >>")
    objs[n_f2] = (b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold "
                  b"/Encoding /WinAnsiEncoding >>")
    for idx, rows in enumerate(pages):
        pnum = page_nums[idx]
        cnum = pnum + 1
        data = content_stream(idx, rows)
        objs[pnum] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PDF_PAGE_W:g} {PDF_PAGE_H:g}] "
            f"/Resources << /Font << /F1 {n_f1} 0 R /F2 {n_f2} 0 R >> >> "
            f"/Contents {cnum} 0 R >>").encode("latin-1")
        objs[cnum] = (f"<< /Length {len(data)} >>\nstream\n".encode("latin-1")
                      + data + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {}
    for num in sorted(objs):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("latin-1") + objs[num] + b"\nendobj\n"
    xref_at = len(out)
    count = max(objs) + 1
    out += f"xref\n0 {count}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for num in range(1, count):
        out += f"{offsets.get(num, 0):010d} 00000 n \n".encode("latin-1")
    out += (f"trailer\n<< /Size {count} /Root 1 0 R >>\nstartxref\n"
            f"{xref_at}\n%%EOF\n").encode("latin-1")

    with open(path, "wb") as f:
        f.write(bytes(out))
    return total


def log_files_for(stream_name, log_dir=LOG_DIR):
    """Every saved log file for a feed, oldest first: [(YYYYMMDD, path), ...]."""
    pattern = os.path.join(log_dir, f"{safe_filename(stream_name)}-*.log")
    out = []
    for p in glob.glob(pattern):
        day = os.path.splitext(os.path.basename(p))[0].rsplit("-", 1)[-1]
        if len(day) == 8 and day.isdigit():
            out.append((day, p))
    return sorted(out)


_LOG_LINE = re.compile(r"^\[(\d{2}:\d{2}:\d{2})\]\s(.*)$")


def parse_log_line(line):
    """Split a saved log line back into (ts, text), or None if it isn't one.
    Logs are written as "[HH:MM:SS] text" by Output.line."""
    m = _LOG_LINE.match(line.rstrip("\n"))
    return (m.group(1), m.group(2)) if m else None


def read_log_lines(paths):
    """Read transcript lines from log files, in the order given. Unreadable files
    are skipped -- a partial export beats no export."""
    lines = []
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                lines.extend(ln.rstrip("\n") for ln in f)
        except OSError:
            continue
    return lines


def day_summaries(stream_name, clips=None, log_dir=LOG_DIR):
    """What's reviewable for a feed, newest day first:
        [{"day", "lines", "clips", "path"}, ...]

    Powers the "open a past day" picker, which has to tell the user two separate
    things: a day can have a transcript but no audio left, because clips are
    purged on a shorter schedule than logs (clips.retention_days vs
    log_retention_days). `clips` is an optional ClipStore used to count surviving
    audio; without one the clips counts are 0."""
    out = []
    for day, path in log_files_for(stream_name, log_dir=log_dir):
        entries = read_log_entries([(day, path)])
        n_clips = 0
        if clips is not None:
            cmap = clips.clip_map(day)
            used = {}
            for _d, ts, _text in entries:
                ids = cmap.get((stream_name, ts), ())
                i = used.get(ts, 0)
                used[ts] = i + 1
                if i < len(ids):
                    n_clips += 1
        out.append({"day": day, "lines": len(entries), "clips": n_clips,
                    "path": path})
    out.sort(key=lambda r: r["day"], reverse=True)      # newest first
    return out


def load_day(stream_name, day, clips=None, log_dir=LOG_DIR):
    """One day's transcript for a feed as [(ts, text, clip_id_or_None)].

    The same second can hold several transmissions, each with its own clip, so
    ids are taken from clip_map in order rather than by lookup -- see
    ClipStore.clip_map. Returns [] if that day has no log."""
    paths = [(d, p) for d, p in log_files_for(stream_name, log_dir=log_dir)
             if d == day]
    if not paths:
        return []
    entries = read_log_entries(paths)
    cmap = clips.clip_map(day) if clips is not None else {}
    used, out = {}, []
    for _d, ts, text in entries:
        ids = cmap.get((stream_name, ts), ())
        i = used.get(ts, 0)
        used[ts] = i + 1
        out.append((ts, text, ids[i] if i < len(ids) else None))
    return out


def read_log_entries(day_paths):
    """Like read_log_lines, but keeps each line's DAY: [(day, ts, text), ...].

    Log files only record the time of day; the date lives in the filename. A PDF
    header that reports the first and last transmission needs both, so callers
    that care about the span read entries instead of bare lines.
    Unparseable lines (a torn write, a stray blank) are skipped."""
    entries = []
    for day, path in day_paths:
        for raw in read_log_lines([path]):
            got = parse_log_line(raw)
            if got:
                entries.append((day, got[0], got[1]))
    return entries


def transcript_span(entries):
    """(first, last) as 'YYYY-MM-DD HH:MM:SS' over [(day, ts, ...)] entries, or
    (None, None) if there are none. Sorted here rather than trusting input order,
    so a caller that concatenates days out of order still gets a true span."""
    stamps = sorted(f"{day[:4]}-{day[4:6]}-{day[6:8]} {ts}"
                    for day, ts, *_ in entries if len(day) == 8)
    return (stamps[0], stamps[-1]) if stamps else (None, None)


def format_transcript_span(first, last):
    """Human-readable span for a transcript header. Collapses the date when the
    whole transcript is from one day, which is the common case:

        2026-07-28  ·  10:38:24 - 11:05:02
        2026-07-12 08:00:01  -  2026-07-19 23:59:12
    """
    if not first and not last:
        return ""
    if not last or first == last:
        return first or last
    d1, t1 = first.split(" ")
    d2, t2 = last.split(" ")
    if d1 == d2:
        return f"{d1}  ·  {t1} – {t2}"
    return f"{first}  –  {last}"


# --------------------------------------------------------------------------
# Clip recording: keep the audio behind each transcript line.
#
# The gate already does the hard part. Gate.push() emits ONE completed segment
# per transmission and that same array is what gets transcribed, so a line and
# its audio are 1:1 -- there is nothing to record continuously and slice up
# later. A clip is just that array written out (preroll_sec included, so it
# doesn't sound chopped).
#
# Storage is the real constraint: 16 kHz mono s16 is 32 KB/s, so a busy feed at
# ~30% duty cycle (~7h of actual transmissions) is ~800 MB/day as WAV. Clips are
# therefore encoded to Opus through the ffmpeg that already decodes the streams
# (no new dependency on either architecture), which measures ~7x smaller on
# scanner-length segments -- call it ~100 MB/day for a busy feed. If ffmpeg can't
# encode Opus we fall back to WAV via the stdlib and say so once.
#
# Writing happens on a worker thread: encoding on the transcribe thread would
# add latency to every line.
# --------------------------------------------------------------------------
CLIP_DIR = os.path.join(DATA_DIR, "clips")
CLIP_QUEUE_MAX = 200          # bounded: never let a stalled disk eat memory
CLIP_CAP_CHECK_EVERY = 100    # writes between size-cap sweeps while running
CLIP_DEFAULTS = {"enabled": False, "retention_days": 7, "bitrate": "24k",
                 "max_gb": 0}          # 0 = no size cap, days only


def clip_settings(cfg):
    """Merge the config's 'clips' block over the defaults."""
    s = dict(CLIP_DEFAULTS)
    s.update(cfg.get("clips") or {})
    return s


def decode_audio_file(path, ffmpeg=None):
    """Decode an audio FILE to raw 16 kHz mono s16 bytes. b"" on any failure."""
    cmd = [ffmpeg or find_ffmpeg(), "-nostdin", "-loglevel", "error", "-i", path,
           "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30,
                              **_no_window_kwargs())
        return proc.stdout if proc.returncode == 0 else b""
    except Exception:
        return b""


def decode_audio_bytes(data, ffmpeg=None):
    """Same, but for audio already in memory -- lets a clip inside a transcript
    bundle play without ever being written to disk."""
    if not data:
        return b""
    cmd = [ffmpeg or find_ffmpeg(), "-nostdin", "-loglevel", "error",
           "-i", "pipe:0", "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
           "pipe:1"]
    try:
        proc = subprocess.run(cmd, input=data, capture_output=True, timeout=30,
                              **_no_window_kwargs())
        return proc.stdout if proc.returncode == 0 else b""
    except Exception:
        return b""


def _f32_to_s16_bytes(audio):
    """Gate segments are float32 in [-1, 1]; clips are int16 PCM."""
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


class ClipStore:
    """Saves and reloads the audio behind transcript lines.

    save() is called from the transcribe thread and returns a clip id
    immediately; the encode happens on this object's writer thread. The id is
    minted up front so the transcript line can carry it before the file exists.
    """

    def __init__(self, cfg, out=None, clip_dir=CLIP_DIR, ffmpeg=None):
        s = clip_settings(cfg)
        self.enabled = bool(s["enabled"])
        self.retention_days = s["retention_days"]
        self.max_gb = s.get("max_gb", 0)
        self.bitrate = s["bitrate"]
        self.dir = clip_dir
        self.out = out
        self.ffmpeg = ffmpeg or find_ffmpeg()
        self._q = queue.Queue(maxsize=CLIP_QUEUE_MAX)
        self._thread = None
        self._stop = threading.Event()
        self._seq = 0
        self._seq_lock = threading.Lock()
        self._opus = None           # None = not probed yet, then True/False
        self._warned = False
        self.dropped = 0            # clips lost to a full queue (writer too slow)
        self._since_cap_check = 0   # writes since the last size-cap sweep

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._thread is not None:
            return
        os.makedirs(self.dir, exist_ok=True)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="clipwriter")
        self._thread.start()

    def stop(self, timeout=5.0):
        """Stop the writer, giving queued clips a chance to land."""
        self._stop.set()
        t = self._thread
        self._thread = None
        if t:
            t.join(timeout=timeout)

    # -- writing -----------------------------------------------------------
    def new_id(self, feed, when=None):
        when = when or dt.datetime.now()
        with self._seq_lock:
            self._seq += 1
            seq = self._seq
        return (f"{safe_filename(feed)}-{when.strftime('%Y%m%d-%H%M%S')}"
                f"-{seq:05d}")

    def save(self, feed, audio, text="", when=None):
        """Queue `audio` (float32 gate segment) for writing. Returns the clip id,
        or None if recording is off or the queue is saturated."""
        if not self.enabled or self._thread is None:
            return None
        when = when or dt.datetime.now()
        clip_id = self.new_id(feed, when)
        item = {
            "id": clip_id, "feed": feed, "day": when.strftime("%Y%m%d"),
            "ts": when.strftime("%H:%M:%S"), "text": text,
            "dur": round(len(audio) / float(SAMPLE_RATE), 2),
        }
        try:
            self._q.put_nowait((item, _f32_to_s16_bytes(audio)))
        except queue.Full:
            # Better to lose a clip than to stall transcription behind the disk.
            self.dropped += 1
            return None
        return clip_id

    def _run(self):
        while not self._stop.is_set() or not self._q.empty():
            try:
                item, pcm = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._write(item, pcm)
            except Exception as e:
                self._warn(f"clip write failed: {e}")
            finally:
                self._q.task_done()

    def _write(self, item, pcm):
        os.makedirs(self.dir, exist_ok=True)
        if self._opus is None:
            self._opus = self._probe_opus()
            if not self._opus:
                self._warn("ffmpeg has no Opus encoder -- saving clips as WAV "
                           "(much larger; consider a shorter clip retention).")
        path = os.path.join(self.dir, item["id"] +
                            (".opus" if self._opus else ".wav"))
        if self._opus:
            self._encode_opus(pcm, path)
        else:
            self._write_wav(pcm, path)
        item["file"] = os.path.basename(path)
        item["bytes"] = os.path.getsize(path)
        with open(self._index_path(item["day"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(item) + "\n")

        # Hold the size cap while running, not only at startup: a session left up
        # for days would otherwise sail past it. Checked periodically rather than
        # per clip -- it's a directory scan, and a hundred clips is a few MB of
        # overshoot at most.
        self._since_cap_check += 1
        if self.max_gb and self._since_cap_check >= CLIP_CAP_CHECK_EVERY:
            self._since_cap_check = 0
            evicted = purge_clips_over_size(float(self.max_gb) * (1 << 30),
                                            self.dir)
            if evicted and self.out:
                self.out.status(f"Clip storage reached {self.max_gb:g} GB — "
                                f"removed the {len(evicted)} oldest clip(s).")

    def _encode_opus(self, pcm, path):
        cmd = [self.ffmpeg, "-nostdin", "-loglevel", "error", "-y",
               "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
               "-c:a", "libopus", "-b:a", self.bitrate, "-application", "voip",
               path]
        proc = subprocess.run(cmd, input=pcm, capture_output=True,
                              **_no_window_kwargs())
        if proc.returncode != 0 or not os.path.exists(path):
            raise RuntimeError((proc.stderr or b"").decode("utf-8", "replace")[-200:]
                               or "ffmpeg failed")

    @staticmethod
    def _write_wav(pcm, path):
        import wave
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm)

    def _probe_opus(self):
        """Does this ffmpeg build have libopus? Probed once, then cached."""
        try:
            proc = subprocess.run([self.ffmpeg, "-hide_banner", "-encoders"],
                                  capture_output=True, timeout=15,
                                  **_no_window_kwargs())
            return b"libopus" in (proc.stdout or b"")
        except Exception:
            return False

    def _warn(self, msg):
        if self.out and not self._warned:
            self._warned = True
            self.out.status(msg)

    # -- reading -----------------------------------------------------------
    def _index_path(self, day):
        return os.path.join(self.dir, f"index-{day}.jsonl")

    def path_for(self, clip_id):
        """Where a clip landed, or None if it isn't on disk (yet)."""
        for ext in (".opus", ".wav"):
            p = os.path.join(self.dir, clip_id + ext)
            if os.path.isfile(p):
                return p
        return None

    def load_pcm(self, clip_id):
        """Decode a clip back to raw 16 kHz mono s16 bytes for playback.
        Returns b"" if the clip is missing or undecodable."""
        path = self.path_for(clip_id)
        if not path:
            return b""
        if path.endswith(".wav"):
            import wave
            try:
                with wave.open(path, "rb") as w:
                    return w.readframes(w.getnframes())
            except Exception:
                return b""
        return decode_audio_file(path, self.ffmpeg)

    def index_for_day(self, day):
        """Every clip record saved on `day` (YYYYMMDD), oldest first."""
        p = self._index_path(day)
        if not os.path.isfile(p):
            return []
        rows = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue        # a torn last line after a hard kill
        return rows

    def find_clip(self, feed, ts, day=None):
        """The clip id for a feed's line at timestamp `ts` (HH:MM:SS), or None.
        Used to re-attach audio to lines restored from disk logs."""
        day = day or dt.datetime.now().strftime("%Y%m%d")
        for row in self.index_for_day(day):
            if row.get("feed") == feed and row.get("ts") == ts:
                return row.get("id")
        return None

    def clip_info(self, clip_id):
        """The index record for one clip, or None. The day is embedded in the id
        ({feed}-YYYYMMDD-HHMMSS-NNNNN), so this reads just that day's index."""
        parts = clip_id.rsplit("-", 3)
        if len(parts) < 4:
            return None
        for row in self.index_for_day(parts[-3]):
            if row.get("id") == clip_id:
                return row
        return None

    def clip_map(self, day=None):
        """{(feed, ts): [clip_id, ...]} for a whole day, read once. Restoring a
        scrollback means thousands of lookups; find_clip() per line would rescan
        the index every time.

        The value is a LIST because timestamps are only second-resolution: a busy
        feed can log several transmissions within one second, and each has its own
        clip. Both the log and the index are chronological, so a caller walking
        the log can take ids from each list in order and keep lines matched to the
        right audio. Collapsing to one id per second would attach the same clip to
        every line in it."""
        out = {}
        for r in self.index_for_day(day or dt.datetime.now().strftime("%Y%m%d")):
            if r.get("feed") and r.get("ts") and r.get("id"):
                out.setdefault((r["feed"], r["ts"]), []).append(r["id"])
        return out


# --------------------------------------------------------------------------
# Transcript bundles: a transcript plus the audio behind it, in one file.
#
# The point is to outlive purge_old_clips -- to keep an incident, or hand it to
# someone else, after the 7-day clip retention has taken the originals. That
# inverts the usual safety property, so it's deliberate and opt-in only.
#
# The container is a plain ZIP with a distinct extension. A private binary format
# would be readable only by this app, which is a poor bet for something whose
# whole purpose is archival: rename a .tscript to .zip and any tool gets the
# transcript as JSON and the audio as ordinary .opus files. Using zipfile also
# keeps this dependency-free on both architectures.
#
#   manifest.json    format/version/app, feed, day, span, counts
#   transcript.json  [{"ts", "text", "clip"}, ...] in order
#   clips/<id>.opus  only the clips the lines reference
#
# Clips are read straight out of the archive and decoded in memory (see
# TranscriptBundle.load_pcm) rather than extracted: these are voice recordings,
# and scattering copies through temp folders to play them would undo the care
# taken everywhere else.
# --------------------------------------------------------------------------
TRANSCRIPT_FORMAT = "transcriber-transcript"
TRANSCRIPT_VERSION = 1
TRANSCRIPT_EXT = ".tscript"


def write_transcript_bundle(path, feed, rows, store, app_version="", day=None):
    """Write a .tscript bundle.

    rows  -- [(ts, text, clip_id_or_None)], the shape load_day returns
    store -- ClipStore (or TranscriptBundle) supplying the audio

    Lines whose clip has already been purged are still written; they simply
    carry no audio, exactly as they appear in the app. Returns
    {"lines": n, "clips": n, "missing": [ids], "bytes": size}."""
    import zipfile

    entries, missing, added = [], [], {}
    for ts, text, clip_id in rows:
        rec = {"ts": ts, "text": text}
        if clip_id:
            src = store.path_for(clip_id) if hasattr(store, "path_for") else None
            data = None
            if src and os.path.isfile(src):
                ext = os.path.splitext(src)[1]
                with open(src, "rb") as f:
                    data = f.read()
            elif hasattr(store, "clip_bytes"):        # re-bundling a bundle
                data, ext = store.clip_bytes(clip_id)
            if data:
                added[clip_id] = (f"clips/{clip_id}{ext}", data)
                rec["clip"] = clip_id
            else:
                missing.append(clip_id)
        entries.append(rec)

    stamps = [r["ts"] for r in entries if r.get("ts")]
    manifest = {
        "format": TRANSCRIPT_FORMAT,
        "version": TRANSCRIPT_VERSION,
        "app_version": app_version,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "feed": feed,
        "day": day or "",
        "lines": len(entries),
        "clips": len(added),
        "first": stamps[0] if stamps else "",
        "last": stamps[-1] if stamps else "",
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
        # The transcript compresses well; the clips are already Opus, so storing
        # them uncompressed saves time and gains nothing to deflate.
        z.writestr("transcript.json", json.dumps(entries, indent=2))
        for name, data in added.values():
            z.writestr(zipfile.ZipInfo(name), data,
                       compress_type=zipfile.ZIP_STORED)
    return {"lines": len(entries), "clips": len(added), "missing": missing,
            "bytes": os.path.getsize(path)}


class TranscriptBundle:
    """A .tscript opened for review. Exposes the same (ts, text, clip_id) rows
    the live views use, and decodes clips from inside the archive.

    Use as a context manager, or call close(). Raises ValueError if the file
    isn't a readable bundle."""

    def __init__(self, path, ffmpeg=None):
        import zipfile
        self.path = path
        self.ffmpeg = ffmpeg or find_ffmpeg()
        try:
            self._zip = zipfile.ZipFile(path, "r")
        except Exception as e:
            raise ValueError(f"Not a readable transcript file: {e}")
        try:
            self.manifest = json.loads(self._zip.read("manifest.json"))
            entries = json.loads(self._zip.read("transcript.json"))
        except KeyError:
            self._zip.close()
            raise ValueError("That file isn't a Transcriber transcript "
                             "(no manifest inside).")
        except Exception as e:
            self._zip.close()
            raise ValueError(f"That transcript file is damaged: {e}")
        if self.manifest.get("format") != TRANSCRIPT_FORMAT:
            self._zip.close()
            raise ValueError("That file isn't a Transcriber transcript.")
        if self.manifest.get("version", 0) > TRANSCRIPT_VERSION:
            self._zip.close()
            raise ValueError(
                f"That transcript was written by a newer version of "
                f"Transcriber (format {self.manifest['version']}). Update the "
                f"app to open it.")

        # Map clip id -> member name from what's actually in the archive, rather
        # than trusting ids from the JSON to build paths.
        self._members = {}
        for name in self._zip.namelist():
            if name.startswith("clips/") and "/" not in name[len("clips/"):]:
                base = os.path.basename(name)
                self._members[os.path.splitext(base)[0]] = name

        self.rows = []
        for e in entries:
            if not isinstance(e, dict):
                continue
            cid = e.get("clip")
            self.rows.append((e.get("ts", ""), e.get("text", ""),
                              cid if cid in self._members else None))

    # -- context manager ---------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        try:
            self._zip.close()
        except Exception:
            pass

    # -- what the viewer needs --------------------------------------------
    @property
    def feed(self):
        return self.manifest.get("feed", "(unknown feed)")

    @property
    def day(self):
        return self.manifest.get("day", "")

    def clip_bytes(self, clip_id):
        """(raw encoded bytes, extension) for a clip, or (None, "")."""
        name = self._members.get(clip_id)
        if not name:
            return None, ""
        try:
            return self._zip.read(name), os.path.splitext(name)[1]
        except Exception:
            return None, ""

    def load_pcm(self, clip_id):
        """Decode a clip to raw PCM without extracting it. Same contract as
        ClipStore.load_pcm, so the player and MP3 export take either source."""
        data, ext = self.clip_bytes(clip_id)
        if not data:
            return b""
        if ext == ".wav":
            import io
            import wave
            try:
                with wave.open(io.BytesIO(data), "rb") as w:
                    return w.readframes(w.getnframes())
            except Exception:
                return b""
        return decode_audio_bytes(data, self.ffmpeg)

    def clip_info(self, clip_id):
        """Minimal record for a clip, so MP3 export can name its output."""
        for ts, text, cid in self.rows:
            if cid == clip_id:
                return {"id": clip_id, "feed": self.feed, "ts": ts,
                        "text": text, "day": self.day}
        return None


MP3_DEFAULT_BITRATE = "64k"     # mono 16 kHz speech; plenty, and small
MP3_GAP_MS = 300                # silence between joined transmissions


def export_clips_mp3(clip_ids, out_path, store, gap_ms=MP3_GAP_MS,
                     bitrate=MP3_DEFAULT_BITRATE, title=None):
    """Decode the given clips, join them in order, and write one MP3.

    Joining happens as raw PCM rather than by concatenating encoded files: the
    clips are Opus and stitching compressed frames would need matching encoder
    state. Decoding to s16 and re-encoding once is simpler and lossless-enough
    for speech at these rates. A short silence separates transmissions so a
    combined file doesn't run together.

    Returns {"clips": n_written, "missing": [ids], "seconds": float}. Raises
    RuntimeError if nothing could be decoded or ffmpeg fails."""
    gap = b"\x00\x00" * int(SAMPLE_RATE * max(0, gap_ms) / 1000)
    chunks, missing = [], []
    for cid in clip_ids:
        pcm = store.load_pcm(cid)
        if not pcm:
            missing.append(cid)
            continue
        if chunks:
            chunks.append(gap)
        chunks.append(pcm)
    if not chunks:
        raise RuntimeError("None of the selected clips could be read "
                           "(they may have been purged).")
    audio = b"".join(chunks)

    cmd = [store.ffmpeg, "-nostdin", "-loglevel", "error", "-y",
           "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0",
           "-c:a", "libmp3lame", "-b:a", bitrate]
    if title:
        cmd += ["-metadata", f"title={title}"]
    cmd += [out_path]
    proc = subprocess.run(cmd, input=audio, capture_output=True,
                          **_no_window_kwargs())
    if proc.returncode != 0 or not os.path.exists(out_path):
        err = (proc.stderr or b"").decode("utf-8", "replace")[-300:]
        raise RuntimeError(err or "ffmpeg failed to write the MP3.")
    n = len(clip_ids) - len(missing)
    return {"clips": n, "missing": missing,
            "seconds": round(len(audio) / 2.0 / SAMPLE_RATE, 2)}


def purge_old_clips(retention_days, clip_dir=CLIP_DIR):
    """Delete clips and index files older than retention_days. Same contract as
    purge_old_logs -- clips are voice recordings, so bounding them matters more,
    not less. retention_days <= 0 disables purging. Returns deleted paths."""
    if not retention_days or retention_days <= 0:
        return []
    if not os.path.isdir(clip_dir):
        return []
    cutoff = time.time() - retention_days * 86400
    deleted = []
    for pattern in ("*.opus", "*.wav", "index-*.jsonl"):
        for path in glob.glob(os.path.join(clip_dir, pattern)):
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    deleted.append(path)
            except OSError:
                pass
    return deleted


def clips_disk_usage(clip_dir=CLIP_DIR):
    """(clip count, total bytes) currently on disk."""
    n = total = 0
    for pattern in ("*.opus", "*.wav"):
        for path in glob.glob(os.path.join(clip_dir, pattern)):
            try:
                total += os.path.getsize(path)
                n += 1
            except OSError:
                pass
    return n, total


def logs_disk_usage(log_dir=LOG_DIR):
    """(file count, total bytes) of saved transcripts. Text is tiny next to the
    audio -- showing both side by side is what makes the separate policies
    obviously right rather than fussy."""
    n = total = 0
    for path in glob.glob(os.path.join(log_dir, "*.log")):
        try:
            total += os.path.getsize(path)
            n += 1
        except OSError:
            pass
    return n, total


def purge_clips_over_size(max_bytes, clip_dir=CLIP_DIR):
    """Delete the OLDEST clips until the folder fits inside max_bytes.

    A day limit alone can't bound the disk: a quiet week and a busy one differ by
    an order of magnitude at the same retention. This is the setting that
    actually caps it. max_bytes <= 0 disables. Returns deleted paths.

    Index files are left alone -- they're tiny, and the day-based purge removes
    them. A clip missing from disk with an index entry still present is already
    handled everywhere (the marker just doesn't render)."""
    if not max_bytes or max_bytes <= 0:
        return []
    if not os.path.isdir(clip_dir):
        return []
    files = []
    for pattern in ("*.opus", "*.wav"):
        for path in glob.glob(os.path.join(clip_dir, pattern)):
            try:
                files.append((os.path.getmtime(path), os.path.getsize(path), path))
            except OSError:
                pass
    total = sum(f[1] for f in files)
    if total <= max_bytes:
        return []
    deleted = []
    for _mtime, size, path in sorted(files):        # oldest first
        if total <= max_bytes:
            break
        try:
            os.remove(path)
            total -= size
            deleted.append(path)
        except OSError:
            pass
    return deleted


def apply_clip_retention(settings, clip_dir=CLIP_DIR):
    """Run both clip policies: age first, then the size cap on what's left.

    Order matters -- expiring by age first means the size cap only has to evict
    clips that are still within their retention window, so it takes the fewest
    files it can. Returns (aged_out, over_size)."""
    aged = purge_old_clips(settings.get("retention_days"), clip_dir)
    cap_gb = settings.get("max_gb") or 0
    over = purge_clips_over_size(float(cap_gb) * (1 << 30), clip_dir) \
        if cap_gb else []
    return aged, over


# --------------------------------------------------------------------------
# Call-sign / unit extraction.
#
# On police/fire radio, units self-identify ("Adam 33", "Engine 14", "Medic 7").
# That spoken call sign is a far more reliable identity cue than any acoustic
# voiceprint on compressed scanner audio, so we color/group by it.
#
# Design = PRECISION over recall: it is better to leave a line uncolored than to
# mislabel a license plate ("King Tom George, 9-0-5-1") or a street address
# ("3658 East 149th") as a unit. We therefore reject those shapes explicitly.
# --------------------------------------------------------------------------

# Phonetic words used as POLICE unit prefixes (NATO + common APCO/department
# names). When several appear in a row they are spelling a plate, not a unit.
PHONETIC_WORDS = {
    # NATO
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "x-ray", "yankee", "zulu",
    # APCO / police department phonetics (Adam-Boy-Charlie family)
    "adam", "boy", "david", "edward", "frank", "george", "henry", "ida",
    "john", "king", "lincoln", "mary", "nora", "ocean", "paul", "queen",
    "robert", "sam", "tom", "union", "william", "young", "zebra", "baker",
    "barney", "ocean", "nora",
}

# Fire / EMS / generic unit designators -> very low plate-confusion risk.
DESIGNATOR_WORDS = {
    "engine", "ladder", "truck", "medic", "ambulance", "rescue", "squad",
    "battalion", "tower", "tanker", "brush", "chief", "ems", "car", "unit",
    "adam",  # also a common police car prefix
}

# Street / address markers: if a number is part of an address, it is NOT a unit.
_ADDRESS_NEXT = re.compile(
    r"^(st|nd|rd|th|street|st\.|ave|avenue|road|rd\.|blvd|boulevard|drive|dr|"
    r"lane|ln|court|ct|place|pl|way|highway|hwy|east|west|north|south)\b", re.I)
_DIR_WORD = {"east", "west", "north", "south"}

# A token is a (possibly hyphenated) word like "x-ray", or a digit group that
# may be spoken with hyphens like "3-1" (= 31) or a plate "9-0-5-1" (= 9051).
_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|\d+(?:-\d+)*")


def _norm_word(w):
    w = w.lower().replace(".", "")
    # Collapse spoken-digit hyphenation ("3-1" -> "31") but keep "x-ray" intact.
    if w and w[0].isdigit():
        w = w.replace("-", "")
    return w


import functools


@functools.lru_cache(maxsize=256)
def _keyword_pattern(kw):
    """Compile a word-boundary regex for one keyword. Cached so repeated calls
    are cheap. Uses \\b boundaries so 'od' doesn't match inside 'understood' and
    'gun' doesn't match inside 'begun'. Multi-word phrases match across spaces."""
    # Escape, then allow flexible whitespace between words of a phrase.
    parts = [re.escape(p) for p in kw.split()]
    body = r"\s+".join(parts)
    return re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE)


def keyword_matches(text, keywords):
    """True if any keyword matches `text` on WORD boundaries (not substrings).
    Handles multi-word phrases ('shots fired') and hyphenated terms ('break-in')."""
    if not text or not keywords:
        return False
    for kw in keywords:
        kw = kw.strip()
        if kw and _keyword_pattern(kw).search(text):
            return True
    return False


# --------------------------------------------------------------------------
# Aviation call signs -> "DELTA 510", "SPEEDBIRD 117", "N65JC".
#
# Same precision-first stance as the police extractor, but a different shape:
# aircraft identify by AIRLINE TELEPHONY NAME + flight number, or by registration
# ("November six five Juliet Charlie"). The police extractor catches Delta only
# because "delta" happens to be in the NATO alphabet it uses for unit prefixes --
# United, Speedbird and the rest get nothing -- so aviation needs its own pass.
#
# The telephony table does double duty: it also yields the airline code used to
# build a FlightRadar24 link for the line.
# --------------------------------------------------------------------------
AIRLINE_TELEPHONY = {
    # US majors / regionals
    "delta": "DL", "united": "UA", "american": "AA", "southwest": "WN",
    "jetblue": "B6", "alaska": "AS", "spirit": "NK", "frontier": "F9",
    "allegiant": "G4", "hawaiian": "HA", "sun country": "SY",
    "envoy": "MQ", "republic": "YX", "endeavor": "9E", "skywest": "OO",
    "piedmont": "PT", "cactus": "AA", "brickyard": "YX",
    # Cargo
    "fedex": "FX", "ups": "5X", "giant": "5Y", "polar": "PO",
    # International
    "speedbird": "BA", "lufthansa": "LH", "air france": "AF", "klm": "KL",
    "shamrock": "EI", "virgin": "VS", "emirates": "EK", "qatari": "QR",
    "cathay": "CX", "japan air": "JL", "all nippon": "NH", "korean air": "KE",
    "singapore": "SQ", "qantas": "QF", "air canada": "AC", "westjet": "WS",
    "aeromexico": "AM", "volaris": "Y4", "iberia": "IB", "alitalia": "AZ",
    "swiss": "LX", "austrian": "OS", "scandinavian": "SK", "finnair": "AY",
    "turkish": "TK", "el al": "LY", "avianca": "AV", "copa": "CM",
    "tam": "JJ", "azul": "AD",
}
_AIRLINE_RE = "|".join(re.escape(k) for k in
                       sorted(AIRLINE_TELEPHONY, key=len, reverse=True))

# NATO letters, for decoding a spoken registration into an N-number.
_NATO_LETTERS = {
    "alpha": "A", "alfa": "A", "bravo": "B", "charlie": "C", "delta": "D",
    "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I",
    "juliet": "J", "juliett": "J", "julia": "J", "kilo": "K", "lima": "L",
    "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q",
    "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V",
    "whiskey": "W", "whisky": "W", "xray": "X", "x-ray": "X", "yankee": "Y",
    "zulu": "Z",
}
_NATO_RE = "|".join(sorted(_NATO_LETTERS, key=len, reverse=True))

# "Delta 510", "Delta 5-10" (Whisper often hyphenates spoken digits),
# "Speedbird 117 heavy", "United 1685".
_AIR_FLIGHT = re.compile(
    r"\b(" + _AIRLINE_RE + r")\s+"
    r"((?:\d[\d\s-]{0,8}\d|\d))"
    r"(?:\s+(?:heavy|super))?\b", re.I)

# "November 65 Juliet Charlie" -> N65JC. US registrations start with November;
# requiring it keeps this from firing on stray phonetics mid-sentence.
_AIR_TAIL = re.compile(
    r"\bnovember\s+((?:(?:\d[\d\s-]{0,6}\d|\d)|(?:" + _NATO_RE + r"))"
    r"(?:[\s,-]+(?:(?:\d[\d\s-]{0,6}\d|\d)|(?:" + _NATO_RE + r"))){0,4})\b",
    re.I)


def _digits_only(s):
    return re.sub(r"\D", "", s)


def extract_aircraft(text):
    """Aircraft mentioned in `text`, in order, as (span, identifier, label).

    span       the text that was matched, for linking it in place
    identifier what FlightRadar24 wants: airline code + flight number ("DL510")
               or a registration ("N65JC")
    label      the canonical display name ("DELTA 510")

    The label is rebuilt from the parts rather than taken from the text, because
    it's also the grouping key for unit colouring and click-to-filter. Whisper
    hyphenates spoken digits unpredictably ("Delta 5-10" one line, "Delta 510"
    the next) and controllers append a weight class ("heavy", "super") that isn't
    part of the identity -- left raw, one aircraft would scatter across several
    labels. Precision over recall, as elsewhere: an unknown telephony word yields
    nothing rather than a guess."""
    if not text:
        return []
    out, taken = [], []

    def _overlaps(a, b):
        return any(not (b <= s or a >= e) for s, e in taken)

    for m in _AIR_FLIGHT.finditer(text):
        num = _digits_only(m.group(2))
        if not 1 <= len(num) <= 4:
            continue
        word = m.group(1).lower()
        code = AIRLINE_TELEPHONY[word]
        out.append((m.start(), m.group(0).strip(), f"{code}{num}",
                    f"{word.upper()} {num}"))
        taken.append((m.start(), m.end()))

    for m in _AIR_TAIL.finditer(text):
        if _overlaps(m.start(), m.end()):
            continue
        ident = "N"
        for tok in re.split(r"[\s,-]+", m.group(1)):
            t = tok.lower()
            if t in _NATO_LETTERS:
                ident += _NATO_LETTERS[t]
            elif t.isdigit():
                ident += t
        # A bare "November" with nothing after it isn't a registration, and real
        # N-numbers are at most 5 characters after the N.
        if 2 <= len(ident) <= 6:
            out.append((m.start(), m.group(0).strip(), ident, ident))

    out.sort()
    return [(span, ident, label) for _pos, span, ident, label in out]


def aircraft_url(identifier):
    """FlightRadar24 page for an aircraft identifier. A registration (N-number)
    resolves to the airframe; anything else is treated as a flight number.

    This only builds a URL for the user's browser -- no API, no scraping."""
    ident = (identifier or "").strip().lower()
    if not ident:
        return ""
    base = "https://www.flightradar24.com/data"
    if re.fullmatch(r"n[0-9][0-9a-z]*", ident):
        return f"{base}/aircraft/{ident}"
    return f"{base}/flights/{ident}"


# --------------------------------------------------------------------------
# Service profiles: what KIND of radio a feed carries.
#
# Three things in this pipeline are domain-specific, not one: the Whisper prompt,
# which call-sign shape to look for, and whether "3658 East 149th" should become
# a map link. Police and Fire/EMS differ only in the prompt -- the call-sign
# extractor already handles both at once (52 police phonetics, 16 fire/EMS
# designators, overlapping on "adam" alone), and shared PD+Fire dispatch channels
# are common, so splitting the extractor would only lose units. ATC is the type
# that genuinely changes behaviour.
#
# A feed with NO service set keeps the historical behaviour exactly: the global
# initial_prompt, emergency call signs, address links on. Nothing migrates.
# --------------------------------------------------------------------------
_POLICE_PROMPT = (
    "The following is police radio dispatch. Common terms: dispatch, copy, "
    "en route, on scene, clear, 10-4, code three, signal, suspect, vehicle, "
    "plate, registration, subject, complainant, requesting backup, be advised, "
    "negative, affirmative, over."
)
_FIRE_PROMPT = (
    "The following is fire and EMS radio dispatch. Common terms: engine, ladder, "
    "truck, medic, ambulance, rescue, squad, battalion, chief, box alarm, "
    "working fire, mutual aid, patient, transport, priority one, on scene, "
    "staging, all clear, copy, be advised, en route."
)
_ATC_PROMPT = (
    "The following is air traffic control radio between controllers and pilots. "
    "Common phraseology: cleared for takeoff, cleared to land, line up and wait, "
    "hold short, taxi via, runway, wind check, contact departure, contact ground, "
    "climb and maintain, descend and maintain, turn left heading, turn right "
    "heading, squawk, ident, traffic in sight, go around, heavy, roger, wilco, "
    "affirm, negative."
)

SERVICE_PRESETS = {
    "police": {"label": "Police", "prompt": _POLICE_PROMPT,
               "callsigns": "emergency", "address_links": True,
               "aircraft_links": False},
    "fire": {"label": "Fire / EMS", "prompt": _FIRE_PROMPT,
             "callsigns": "emergency", "address_links": True,
             "aircraft_links": False},
    "atc": {"label": "Air traffic control", "prompt": _ATC_PROMPT,
            "callsigns": "aviation", "address_links": False,
            "aircraft_links": True},
    "general": {"label": "General", "prompt": "",
                "callsigns": None, "address_links": False,
                "aircraft_links": False},
}

# What a feed with no service set does -- i.e. every feed that existed before
# service profiles were added.
SERVICE_DEFAULT = {"label": "Police / Fire-EMS (default)", "prompt": None,
                   "callsigns": "emergency", "address_links": True,
                   "aircraft_links": False}


def service_profile(stream, cfg=None):
    """Resolve how a feed should be transcribed and rendered.

    Returns a dict with: service, label, prompt, callsigns, address_links,
    aircraft_links. The prompt resolves per-feed override -> service preset ->
    the global initial_prompt, so a feed can always be tuned without touching
    the others."""
    stream = stream or {}
    name = (stream.get("service") or "").lower() or None
    preset = SERVICE_PRESETS.get(name, SERVICE_DEFAULT)
    prompt = stream.get("initial_prompt")          # per-feed override
    if not prompt:
        prompt = preset["prompt"]
    if prompt is None:                             # default profile: use global
        prompt = (cfg or {}).get("initial_prompt") or ""
    out = dict(preset)
    out["service"] = name
    out["prompt"] = prompt
    return out


def extract_callsign(text, extra_prefixes=None, style="emergency"):
    """
    Return a normalized unit call sign found in `text` (e.g. "ADAM 33",
    "ENGINE 14") or None. High precision: rejects spelled-out plates and street
    addresses. `extra_prefixes` (iterable of lowercase words) extends the set of
    recognized unit prefixes for local department lingo.

    `style` selects the domain: "emergency" (police/fire units, the default) or
    "aviation" (airline flights and registrations). A feed's service profile
    picks this -- see SERVICE_PRESETS.

    NOTE: this identifies the FIRST unit MENTIONED in a transmission, which is a
    heuristic for who is involved -- not a guaranteed acoustic speaker ID.
    """
    if not text:
        return None
    if style == "aviation":
        found = extract_aircraft(text)
        return found[0][2] if found else None
    if style in (None, "none"):
        return None
    prefixes = set(PHONETIC_WORDS) | set(DESIGNATOR_WORDS)
    if extra_prefixes:
        prefixes |= {p.lower() for p in extra_prefixes}

    tokens = _WORD_RE.findall(text)
    norm = [_norm_word(t) for t in tokens]

    for i, w in enumerate(norm):
        if w not in prefixes:
            continue
        # Need a following number token.
        if i + 1 >= len(norm) or not norm[i + 1].isdigit():
            continue
        num = norm[i + 1]

        # --- reject plate-spelling: phonetic word adjacent to another phonetic
        # word (e.g. "King Tom George ..."). Designators (Engine/Medic) exempt.
        is_phonetic = w in PHONETIC_WORDS and w not in DESIGNATOR_WORDS
        if is_phonetic:
            prev_ph = i > 0 and norm[i - 1] in PHONETIC_WORDS
            next_ph = i + 1 < len(norm) and norm[i + 1] in PHONETIC_WORDS
            if prev_ph or next_ph:
                continue
            # Plates spoken digit-by-digit show up as 1-digit tokens in a row;
            # a real unit number is 1-3 digits as a single token.
            if len(num) > 3:
                continue
        else:
            if len(num) > 4:  # designator units can be up to 4 digits
                continue

        # --- reject addresses: "<number> East/149th/Street ..."
        nxt = " ".join(tokens[i + 2:i + 3]) if i + 2 < len(tokens) else ""
        if nxt and _ADDRESS_NEXT.match(nxt):
            continue
        # Prefix itself is a direction word followed by a number -> address-ish.
        if w in _DIR_WORD:
            continue

        return f"{w.upper()} {num}"
    return None


# --------------------------------------------------------------------------
# Address / street extraction -> clickable Google-Maps links in the transcript.
#
# Aggressive detection (catch numbered addresses, "Street/Ave/Blvd" mentions,
# and "X and Y" intersections) but guarded against the many false "X and Y"
# English phrases on scanner audio ("conscious and alert", "salt and pepper").
# --------------------------------------------------------------------------
STREET_TYPES = (
    "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd",
    "drive", "dr", "lane", "ln", "court", "ct", "place", "pl", "way",
    "circle", "cir", "parkway", "pkwy", "highway", "hwy", "terrace", "trail",
    "square", "sq", "route", "rt", "expressway",
)
_STREET_TYPE_RE = "|".join(sorted(STREET_TYPES, key=len, reverse=True))
_DIRS = r"(?:north|south|east|west|n|s|e|w|northeast|northwest|southeast|southwest|ne|nw|se|sw)"

# A street "name" token: a capitalized word, an ordinal (149th, 5th), or a
# direction. Numbers-with-ordinal count as street names ("East 149th").
_NAME = r"(?:[A-Z][a-zA-Z]+|\d{1,3}(?:st|nd|rd|th)|" + _DIRS + r")"
# A name word that is NOT a street type (so the type terminates the name and
# isn't swallowed as another name word, which would then grab trailing junk).
_NAME_NT = r"(?!(?i:" + _STREET_TYPE_RE + r")\b)" + _NAME

# 1) Numbered street address: "3658 East 149th Street", "162 America Boulevard",
#    "66745 Schubert Drive". Number + 1-3 name words + optional street type.
#
# NOT compiled with re.I. _NAME requires a capitalised word, and a blanket
# re.I silently defeated that -- every lowercase word became a candidate street
# name, so "Engine 14 show me en route" parsed as 14 + "show me en" + the street
# type "route" and put a Google Maps link on it. ("en route" is about as common
# as radio traffic gets.) Only the direction and street-type alternations are
# case-insensitive, matching how _ADDR_NAMED already does it.
_ADDR_NUMBERED = re.compile(
    r"\b(\d{2,6})\s+"
    r"((?:(?i:" + _DIRS + r")\s+)?" + _NAME_NT + r"(?:\s+" + _NAME_NT + r"){0,2})"
    r"(?:\s+((?i:" + _STREET_TYPE_RE + r")))?\b")

# 2) Named street with an explicit type: "American Boulevard", "Schubert Drive",
#    "Detroit Road". Name(s) immediately followed by a street type word.
#    The NAME stays case-sensitive (requires a capitalized proper noun); the
#    street type is case-insensitive via inline (?i:...).
_ADDR_NAMED = re.compile(
    r"\b((?:(?i:" + _DIRS + r")\s+)?[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+"
    r"((?i:" + _STREET_TYPE_RE + r"))\b")

# 3) Intersection: "Detroit and Dover", "Revere and Butternut", "East 93rd and
#    Union". Both sides must look like street names AND not be common words.
#    A side is: optional direction + (ordinal | capitalized name).
_XNAME = r"(?:(?i:" + _DIRS + r")\s+)?(?:\d{1,3}(?:st|nd|rd|th)|[A-Z][a-zA-Z]+)"
_ADDR_INTERSECTION = re.compile(
    r"\b(" + _XNAME + r")\s+and\s+(" + _XNAME + r")\b")

# Words that frequently appear in "X and Y" but are NOT streets -- reject an
# intersection match if either side is one of these.
_NOT_STREET_WORDS = {
    "conscious", "alert", "responsive", "unresponsive", "salt", "pepper",
    "black", "white", "male", "female", "vehicles", "vehicle", "signs",
    "sneakers", "sneakers", "vans", "shoes", "shirt", "hair", "eyes",
    "blue", "red", "green", "gray", "grey", "orange", "brown", "clear",
    "over", "out", "up", "down", "here", "there", "him", "her", "them",
    "again", "appreciate", "quarter", "quarters", "time", "everyone",
    "everybody", "sir", "again", "advised", "copy", "aware", "safe",
    "sound", "fire", "ems", "fine", "okay", "good", "well",
}


def _looks_like_street(token):
    """A single street-name token (may be multi-word direction+name)."""
    words = token.strip().split()
    core = words[-1].lower().rstrip(".,")
    if core in _NOT_STREET_WORDS:
        return False
    # Ordinals (149th) and capitalized proper names qualify.
    if re.match(r"\d{1,3}(st|nd|rd|th)$", core):
        return True
    return words[-1][:1].isupper()


def extract_addresses(text):
    """Return a list of (matched_span_text, map_query) for addresses/streets found
    in `text`, in order, non-overlapping. `map_query` is the cleaned string to
    hand to a maps search (without city; the GUI appends per-feed city). Aggressive
    but guards common non-street 'X and Y' phrases."""
    if not text:
        return []
    found = []
    claimed = []   # (start, end) spans already taken, to avoid overlaps

    def overlaps(s, e):
        return any(not (e <= cs or s >= ce) for cs, ce in claimed)

    def add(m, query):
        s, e = m.start(), m.end()
        if overlaps(s, e):
            return
        claimed.append((s, e))
        found.append((text[s:e].strip(), " ".join(query.split())))

    # 1) Numbered addresses (highest confidence).
    for m in _ADDR_NUMBERED.finditer(text):
        num, name, stype = m.group(1), m.group(2), m.group(3)
        # Require either a street type OR an ordinal name to avoid grabbing
        # "unit 306" or "710 mail" style false hits.
        if stype or re.search(r"\d{1,3}(st|nd|rd|th)\b", name, re.I) \
                or name.split()[-1][:1].isupper():
            q = f"{num} {name}" + (f" {stype}" if stype else "")
            add(m, q)

    # 2) Named streets with explicit type.
    for m in _ADDR_NAMED.finditer(text):
        name = m.group(1)
        if name.split()[-1].lower() not in _NOT_STREET_WORDS:
            add(m, f"{name} {m.group(2)}")

    # 3) Intersections -- both sides must look like streets.
    for m in _ADDR_INTERSECTION.finditer(text):
        a, b = m.group(1), m.group(2)
        if _looks_like_street(a) and _looks_like_street(b):
            add(m, f"{a} and {b}")

    # Return in order of appearance.
    found_sorted = sorted(found, key=lambda f: text.find(f[0]))
    return found_sorted


def maps_url(query, location=None):
    """Build a Google Maps search URL for `query`, optionally anchored to a city
    (e.g. 'Cleveland, OH') so bare street names resolve to the right place."""
    import urllib.parse
    q = query if not location else f"{query}, {location}"
    return "https://www.google.com/maps/search/?api=1&query=" + \
        urllib.parse.quote(q)


def purge_old_logs(retention_days, log_dir=LOG_DIR):
    """
    Delete *.log files in log_dir older than retention_days (by modified time).
    retention_days <= 0 (or None) disables purging. Returns list of deleted paths.
    Since logs contain sensitive PII, this keeps the on-disk footprint bounded.
    """
    if not retention_days or retention_days <= 0:
        return []
    if not os.path.isdir(log_dir):
        return []
    cutoff = time.time() - retention_days * 86400
    deleted = []
    for path in glob.glob(os.path.join(log_dir, "*.log")):
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
                deleted.append(path)
        except OSError:
            pass
    return deleted


# --------------------------------------------------------------------------
# Update check: compare the installed transcription-engine packages against the
# latest on PyPI. Pure stdlib (urllib), short timeout, fails silently offline.
# Reports only -- it never installs anything (updating is a deliberate, manual
# `pip install -U ...` step, to avoid re-triggering the Python-version wheel trap).
# Which packages matter depends on the engine: the ct2 ones aren't even installed
# on ARM, where whisper.cpp/pywhispercpp is what's actually running.
# --------------------------------------------------------------------------
UPDATE_PACKAGES = ["faster-whisper", "ctranslate2"]      # ct2 / x64
UPDATE_PACKAGES_WHISPERCPP = ["pywhispercpp"]            # whisper.cpp / ARM


def update_packages(cfg=None):
    """The packages worth version-checking for the active engine."""
    return (list(UPDATE_PACKAGES_WHISPERCPP)
            if select_backend(cfg or {}) == "whispercpp" else list(UPDATE_PACKAGES))


def installed_version(pkg):
    try:
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


def _parse_version(v):
    """Best-effort PEP440-ish tuple for comparison, e.g. '1.2.10' -> (1,2,10)."""
    parts = []
    for chunk in str(v).split("."):
        m = re.match(r"\d+", chunk)
        parts.append(int(m.group()) if m else 0)
    return tuple(parts)


def is_newer(latest, current):
    """True if latest > current (numeric, zero-padded)."""
    if not latest or not current:
        return False
    a, b = _parse_version(latest), _parse_version(current)
    n = max(len(a), len(b))
    a += (0,) * (n - len(a))
    b += (0,) * (n - len(b))
    return a > b


def _pypi_latest(pkg, timeout=4.0):
    """Latest version string for a package from PyPI, or None on any failure."""
    import urllib.request
    url = f"https://pypi.org/pypi/{pkg}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("info", {}).get("version")
    except Exception:
        return None


def check_for_updates(packages=None, timeout=4.0, cfg=None):
    """
    Return a list of dicts, one per package:
      {"package", "installed", "latest", "update_available"}
    'latest' is None if PyPI couldn't be reached (offline / blocked).
    `packages` defaults to the active engine's packages (see update_packages).
    """
    if packages is None:
        packages = update_packages(cfg)
    results = []
    for pkg in packages:
        cur = installed_version(pkg)
        latest = _pypi_latest(pkg, timeout=timeout)
        results.append({
            "package": pkg,
            "installed": cur,
            "latest": latest,
            "update_available": is_newer(latest, cur),
        })
    return results


# --------------------------------------------------------------------------
# App self-update: check GitHub Releases for a newer Transcriber build.
# (Separate from check_for_updates() above, which reports ML-library versions
# from PyPI and never installs anything.) The repo is public, so these calls
# are anonymous — no token needed.
# --------------------------------------------------------------------------
APP_REPO = "DevCon-Productions/Transcriber"


def _pick_installer_asset(assets):
    """Choose the installer asset (.exe) matching THIS build's architecture.

    Releases carry both x64 (`Transcriber-Setup-<v>.exe`) and ARM64
    (`Transcriber-ARM64-Setup-<v>.exe`) installers. This is the ARM build, so it
    must pick the arm64-named asset and NEVER fall back to the x64 one (installing
    the wrong architecture). If no arm64 installer is on the release yet, return
    None -> the updater simply reports no update rather than downloading x64.
    (The x64 build makes the mirror choice: the .exe whose name does NOT contain
    'arm64'.)"""
    want_arm = interpreter_is_arm64()
    for a in assets:
        name = str(a.get("name", "")).lower()
        if not name.endswith(".exe"):
            continue
        is_arm_asset = "arm64" in name or "-arm-" in name
        if want_arm == is_arm_asset:
            return a
    return None


def check_for_app_update(current_version, repo=APP_REPO, timeout=6.0):
    """Query the repo's latest GitHub Release and compare it to the running
    version. Returns a dict or None (on any failure — offline, rate-limited, no
    installer asset):
      {available, current, latest, notes, html_url,
       asset_name, asset_url, asset_size}
    """
    import urllib.request
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    try:
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Transcriber-Updater",
        })
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    tag = (data.get("tag_name") or "").lstrip("vV")
    if not tag:
        return None
    asset = _pick_installer_asset(data.get("assets", []))
    return {
        "available": is_newer(tag, current_version),
        "current": current_version,
        "latest": tag,
        "notes": data.get("body") or "",
        "html_url": data.get("html_url") or "",
        "asset_name": asset.get("name") if asset else None,
        "asset_url": asset.get("browser_download_url") if asset else None,
        "asset_size": asset.get("size") if asset else None,
    }


def download_file(url, dest, progress_cb=None, chunk=1 << 20, timeout=30.0):
    """Stream-download `url` to `dest`, writing to a .part file and renaming on
    success. Calls progress_cb(bytes_done, total_bytes) as it goes (total is 0
    if the server sends no Content-Length). Returns dest; raises on failure."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Transcriber-Updater"})
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        with open(tmp, "wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if progress_cb:
                    progress_cb(done, total)
    os.replace(tmp, dest)
    return dest


# Match a Broadcastify player-page URL or bare feed id -> capture the feed id.
_BCFY_PAGE = re.compile(r"broadcastify\.com/(?:listen/)?feed/(\d+)", re.I)
_BARE_ID = re.compile(r"^\d+$")


def normalize_url(url, provider=None):
    """
    Turn a Broadcastify player-page URL (or a bare numeric feed id) into the
    direct, capturable audio stream URL. Non-Broadcastify URLs pass through
    unchanged so any other Icecast/HTTP stream still works.
    """
    url = url.strip()
    feed_id = None
    if provider == "broadcastify" and _BARE_ID.match(url):
        feed_id = url
    else:
        m = _BCFY_PAGE.search(url)
        if m:
            feed_id = m.group(1)
    if feed_id:
        return f"https://audio.broadcastify.com/{feed_id}.mp3"
    return url


# The values the installer seeds into a fresh credentials.json (from
# credentials.example.json). Treated as "not configured" so the app doesn't try
# to authenticate with literal placeholder text (which just makes feeds drop).
_PLACEHOLDER_CREDS = {"YOUR_BROADCASTIFY_USERNAME", "YOUR_BROADCASTIFY_PASSWORD"}


def _clean_cred(v):
    """Return a usable credential string, or None for blank/placeholder values."""
    if not v:
        return None
    v = v.strip()
    if not v or v in _PLACEHOLDER_CREDS:
        return None
    return v


def load_credentials(cfg=None):
    """
    Resolve Broadcastify Premium credentials, in priority order:
      1. credentials.json  ({"broadcastify": {"username": "...", "password": "..."}})
      2. env vars BROADCASTIFY_USERNAME / BROADCASTIFY_PASSWORD
    Placeholder/blank values are ignored. Returns (username, password) or
    (None, None) if not configured.
    """
    user = pw = None
    if os.path.exists(CREDENTIALS_PATH):
        try:
            with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
                creds = json.load(f).get("broadcastify", {})
            user, pw = creds.get("username"), creds.get("password")
        except Exception:
            pass
    user = _clean_cred(user) or _clean_cred(os.environ.get("BROADCASTIFY_USERNAME"))
    pw = _clean_cred(pw) or _clean_cred(os.environ.get("BROADCASTIFY_PASSWORD"))
    return user, pw


def save_credentials(username, password):
    """Write Broadcastify credentials to credentials.json (creating it if
    needed). Preserves any other top-level keys already in the file. Returns
    True on success."""
    data = {}
    if os.path.exists(CREDENTIALS_PATH):
        try:
            with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    data["broadcastify"] = {"username": (username or "").strip(),
                            "password": (password or "").strip()}
    try:
        os.makedirs(os.path.dirname(CREDENTIALS_PATH), exist_ok=True)
        with open(CREDENTIALS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def credentials_configured():
    """True if usable (non-placeholder) Broadcastify credentials are available."""
    user, pw = load_credentials()
    return bool(user and pw)


def is_broadcastify_stream(stream):
    """True if a stream is a Broadcastify feed (needs Premium auth)."""
    return (stream.get("provider") == "broadcastify"
            or "broadcastify.com" in (stream.get("url") or ""))


def enable_windows_ansi():
    """Enable ANSI color escapes in the Windows console (Win10+)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.SetConsoleMode(k.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


# --------------------------------------------------------------------------
# Output: thread-safe sink for transcript lines and status messages.
#
# Always handles per-stream file logging (unless disabled). Optional callbacks
# let a GUI (or any other front-end) receive the same events; the default CLI
# behaviour prints to the console with ANSI colors.
# --------------------------------------------------------------------------
class Output:
    def __init__(self, on_line=None, on_status=None, console=True, file_logging=True):
        self._lock = threading.Lock()
        # callback(stream_name, color, text, ts, clip_id); clip_id is None unless
        # this feed is recording and the clip was queued successfully.
        self.on_line = on_line
        self.on_status = on_status      # callback(msg)
        self.console = console
        self.file_logging = file_logging
        if self.file_logging:
            os.makedirs(LOG_DIR, exist_ok=True)

    def line(self, stream_name, color, text, clip_id=None, when=None):
        # `when` lets the caller pin the line and its saved clip to the SAME
        # instant -- otherwise the two can straddle a second boundary and the
        # clip index no longer matches the timestamp written to the log.
        when = when or dt.datetime.now()
        ts = when.strftime("%H:%M:%S")
        with self._lock:
            if self.console:
                code = ANSI_COLORS.get(color, "97")
                print(f"\033[{code}m[{ts}] {stream_name:<10}\033[0m {text}", flush=True)
            if self.file_logging:
                day = when.strftime("%Y%m%d")
                path = os.path.join(LOG_DIR, f"{safe_filename(stream_name)}-{day}.log")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {text}\n")
        if self.on_line:
            try:
                self.on_line(stream_name, color, text, ts, clip_id)
            except Exception:
                pass

    def status(self, msg):
        with self._lock:
            if self.console:
                print(f"\033[90m{msg}\033[0m", flush=True)
        if self.on_status:
            try:
                self.on_status(msg)
            except Exception:
                pass


# --------------------------------------------------------------------------
# Audio playback: plays the raw PCM of ONE selected stream through the speakers.
#
# Workers always feed their decoded PCM here tagged with their stream name; the
# player only emits audio for the currently-selected source (listen one-at-a-
# time). sounddevice is imported lazily so the headless CLI never depends on it.
#
# A saved clip (play_clip) takes priority over the live feed rather than mixing
# with it: two radio voices at once is unlistenable, and the point of clicking a
# line is to hear THAT transmission. Live audio resumes when the clip ends.
# --------------------------------------------------------------------------
class AudioPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._source = None             # name of the stream currently audible
        self._stream = None
        self._sd = None
        self._buf = bytearray()
        self._clip = bytearray()        # one-shot clip, drained before _buf
        self._ok = self._init_device()

    def _init_device(self):
        try:
            import sounddevice as sd
            self._sd = sd
            self._stream = sd.RawOutputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                blocksize=0, callback=self._callback,
            )
            self._stream.start()
            return True
        except Exception:
            return False

    @property
    def available(self):
        return self._ok

    def _callback(self, outdata, frames, time_info, status):
        need = frames * 2  # int16 mono -> 2 bytes/frame
        with self._lock:
            src = self._clip if self._clip else self._buf
            have = min(need, len(src))
            outdata[:have] = bytes(src[:have])
            del src[:have]
            # While a clip plays, live audio is discarded rather than queued --
            # otherwise the feed would blast a backlog the moment the clip ends.
            if self._clip:
                self._buf.clear()
        if have < need:
            outdata[have:] = b"\x00" * (need - have)

    def set_source(self, name):
        """Select which stream is audible. None mutes everything."""
        with self._lock:
            self._source = name
            self._buf.clear()           # drop buffered audio from the old source

    def get_source(self):
        with self._lock:
            return self._source

    def play_clip(self, pcm_bytes):
        """Play a saved clip once, over the top of whatever is live. Replaces any
        clip already playing (clicking a second line interrupts the first)."""
        if not self._ok or not pcm_bytes:
            return False
        with self._lock:
            self._clip = bytearray(pcm_bytes)
            self._buf.clear()
        return True

    def stop_clip(self):
        """Cut a clip short and hand the speakers back to the live feed."""
        with self._lock:
            self._clip.clear()

    def clip_playing(self):
        with self._lock:
            return bool(self._clip)

    def feed(self, name, pcm_bytes):
        if not self._ok:
            return
        with self._lock:
            if name != self._source or self._clip:
                return
            # Guard against unbounded growth if the device stalls (~2s cap).
            if len(self._buf) > SAMPLE_RATE * 2 * 2:
                self._buf.clear()
            self._buf.extend(pcm_bytes)

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Text-to-speech: reads selected transcript lines aloud. A background thread
# pulls text off a queue and synthesizes+plays it one utterance at a time so
# nothing overlaps. Two engines (Piper neural voices, or Windows SAPI5 which is
# native on ARM64); selected at runtime. Lazy so the app runs fine without TTS.
# Speaks via sounddevice (separate stream from AudioPlayer).
# --------------------------------------------------------------------------
# Two synthesis engines, chosen at runtime:
#   * Piper (neural, .onnx voices) -- default on x64; needs a compiled espeak-ng
#     phonemizer that has NO Windows-ARM64 build.
#   * Windows SAPI5 (via comtypes) -- native everywhere on Windows incl. ARM64,
#     no compilation; uses the OS voices (e.g. Microsoft David / Zira).
# The app calls list_tts_voices()/tts_available() (engine-aware) and TTSPlayer;
# the synth backend is created ON the TTS thread (SAPI is COM -> single-threaded).
def _piper_voices():
    """[(voice_id, path)] of downloaded Piper voices (*.onnx). voice_id is the
    filename stem (e.g. 'en_US-lessac-medium')."""
    out = []
    if os.path.isdir(TTS_VOICE_DIR):
        for p in sorted(glob.glob(os.path.join(TTS_VOICE_DIR, "*.onnx"))):
            out.append((os.path.splitext(os.path.basename(p))[0], p))
    return out


def _piper_usable():
    """True only if Piper can actually synthesize here: importable, a voice
    present, AND its compiled espeak-ng phonemizer available (absent on ARM64)."""
    try:
        import piper  # noqa: F401
    except Exception:
        return False
    if not _piper_voices():
        return False
    import importlib.util
    return (importlib.util.find_spec("piper.espeakbridge") is not None
            or importlib.util.find_spec("piper_phonemize") is not None)


def _sapi_voices():
    """[(name, name)] of installed Windows SAPI5 voices (empty off-Windows / on
    failure). Creates a transient COM object; released immediately."""
    try:
        import comtypes.client
        toks = comtypes.client.CreateObject("SAPI.SpVoice").GetVoices()
        return [(toks.Item(i).GetDescription(), toks.Item(i).GetDescription())
                for i in range(toks.Count)]
    except Exception:
        return []


def _sapi_usable():
    return len(_sapi_voices()) > 0


def _winrt_voices():
    """[(name, name)] of ALL installed Windows voices via WinRT/OneCore. This is a
    superset of classic SAPI5, which only reads the legacy registry hive -- e.g.
    'Microsoft Mark' is present here but invisible to SAPI."""
    try:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        return [(v.display_name, v.display_name)
                for v in SpeechSynthesizer.all_voices]
    except Exception:
        return []


def _winrt_usable():
    return len(_winrt_voices()) > 0


def select_tts_engine(cfg_tts=None):
    """'piper', 'winrt' or 'sapi'. Explicit tts['engine'] wins; 'auto' (default)
    prefers Piper where it can synthesize, then WinRT/OneCore (sees every installed
    Windows voice), then classic SAPI5 as a last resort."""
    pref = str((cfg_tts or {}).get("engine", "auto")).strip().lower()
    if pref in ("piper", "winrt", "sapi"):
        return pref
    if _piper_usable():
        return "piper"
    if _winrt_usable():
        return "winrt"
    if _sapi_usable():
        return "sapi"
    return "piper"


def available_tts_engines():
    """TTS engine ids that can actually speak on this system, best-first
    (piper, winrt, sapi). Used by the GUI to offer only working engines."""
    out = []
    if _piper_usable():
        out.append("piper")
    if _winrt_usable():
        out.append("winrt")
    if _sapi_usable():
        out.append("sapi")
    return out


def list_tts_voices(engine=None, cfg_tts=None):
    """Voices for the selected engine as [(voice_id, detail)]. Piper -> (stem,
    path); WinRT/SAPI -> (display name, display name)."""
    engine = engine or select_tts_engine(cfg_tts)
    if engine == "winrt":
        return _winrt_voices()
    if engine == "sapi":
        return _sapi_voices()
    return _piper_voices()


def tts_available(cfg_tts=None):
    """True if the selected TTS engine can actually speak on this system."""
    engine = select_tts_engine(cfg_tts)
    if engine == "winrt":
        return _winrt_usable()
    if engine == "sapi":
        return _sapi_usable()
    return _piper_usable()


def _match_voice_name(voice_id, names):
    """Best match for a saved voice id among `names`, or None. Exact, then case-
    insensitive, then a loose prefix match so a voice saved under one engine still
    resolves under another (SAPI's 'Microsoft Zira Desktop - English (United
    States)' -> WinRT's 'Microsoft Zira')."""
    if not voice_id or not names:
        return None
    if voice_id in names:
        return voice_id
    low = voice_id.strip().lower()
    for n in names:
        if n.strip().lower() == low:
            return n
    for n in names:                     # engine naming differs -> prefix match
        nl = n.strip().lower()
        if low.startswith(nl) or nl.startswith(low):
            return n
    return None


# -- synth backends: built and used on the TTS thread; expose synthesize(text)
#    -> int16 mono np.ndarray and a `sample_rate` / `voice_id`. -----------------
class _PiperSynth:
    engine = "piper"

    def __init__(self, voice_id):
        from piper import PiperVoice
        voices = dict(_piper_voices())
        if not voices:
            raise RuntimeError("no Piper voice models in tts_voices/")
        self.voice_id = voice_id if voice_id in voices else next(iter(voices))
        self._voice = PiperVoice.load(voices[self.voice_id])
        self.sample_rate = self._voice.config.sample_rate

    def synthesize(self, text):
        chunks = [np.frombuffer(c.audio_int16_bytes, dtype=np.int16)
                  for c in self._voice.synthesize(text)]
        return np.concatenate(chunks) if chunks else np.zeros(0, np.int16)


class _SapiSynth:
    engine = "sapi"
    sample_rate = 16000                       # SAFT16kHz16BitMono -> matches pipeline

    def __init__(self, voice_id):
        import comtypes.client
        self._ct = comtypes.client
        self._voice = comtypes.client.CreateObject("SAPI.SpVoice")
        from comtypes.gen import SpeechLib     # generated by the CreateObject above
        self._fmt_type = SpeechLib.SAFT16kHz16BitMono
        toks = self._voice.GetVoices()
        if toks.Count == 0:
            raise RuntimeError("no Windows SAPI voices installed")
        want = _match_voice_name(
            voice_id, [toks.Item(i).GetDescription() for i in range(toks.Count)])
        chosen = toks.Item(0)                 # default if the saved voice is gone
        for i in range(toks.Count):
            if toks.Item(i).GetDescription() == want:
                chosen = toks.Item(i)
                break
        self._voice.Voice = chosen
        self.voice_id = chosen.GetDescription()

    def synthesize(self, text):
        stream = self._ct.CreateObject("SAPI.SpMemoryStream")
        fmt = self._ct.CreateObject("SAPI.SpAudioFormat")
        fmt.Type = self._fmt_type
        stream.Format = fmt
        self._voice.AudioOutputStream = stream
        self._voice.Speak(text, 0)            # 0 = SVSFDefault (synchronous)
        return np.frombuffer(bytes(stream.GetData()), dtype=np.int16)


class _WinrtSynth:
    """Windows OneCore speech via WinRT. Sees EVERY installed Windows voice (classic
    SAPI5 only reads the legacy hive) and needs no registry mirroring -- so voices
    added via Settings > Time & language > Speech show up here. Synthesizes to a WAV
    stream and returns its PCM."""
    engine = "winrt"

    def __init__(self, voice_id):
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer
        voices = SpeechSynthesizer.all_voices
        if len(voices) == 0:
            raise RuntimeError("no Windows (WinRT) voices installed")
        self._synth = SpeechSynthesizer()
        want = _match_voice_name(voice_id, [v.display_name for v in voices])
        for v in voices:                      # keep the default voice if no match
            if v.display_name == want:
                self._synth.voice = v
                break
        self.voice_id = self._synth.voice.display_name
        self.sample_rate = 16000              # refreshed from each WAV header

    async def _to_wav(self, text):
        from winrt.windows.storage.streams import DataReader
        stream = await self._synth.synthesize_text_to_stream_async(text)
        size = stream.size
        reader = DataReader(stream.get_input_stream_at(0))
        await reader.load_async(size)
        buf = bytearray(size)
        reader.read_bytes(buf)                # fills the caller's buffer
        return bytes(buf)

    def synthesize(self, text):
        import asyncio
        import io
        import wave
        with wave.open(io.BytesIO(asyncio.run(self._to_wav(text))), "rb") as w:
            self.sample_rate = w.getframerate()
            channels = w.getnchannels()
            frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:                      # downmix to mono
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
        return audio


def _make_tts_synth(engine, voice_id):
    if engine == "winrt":
        return _WinrtSynth(voice_id)
    if engine == "sapi":
        return _SapiSynth(voice_id)
    return _PiperSynth(voice_id)


class TTSPlayer(threading.Thread):
    """Background speech queue: put(text) -> spoken aloud, one at a time.
    Drops items if the backlog grows (so it never lags far behind live audio).
    Optional on_start(text)/on_end(text) callbacks fire around each utterance
    (used by the GUI to highlight the line currently being read)."""
    def __init__(self, voice_id=None, max_queue=6, out=None,
                 on_start=None, on_end=None, engine=None):
        self.on_start = on_start
        self.on_end = on_end
        super().__init__(daemon=True, name="tts")
        self.q = queue.Queue(maxsize=max_queue)
        self.stop_evt = threading.Event()
        self.out = out
        self._engine = engine or select_tts_engine()
        self._voice_id = voice_id
        self._synth = None                       # built on the TTS thread in run()
        self._sr = 22050
        self._ok = tts_available({"engine": self._engine})
        self._muted = False
        self._speaking = False          # an utterance is actually sounding
        self._cancel = False            # one-shot: stop the current utterance

    @property
    def available(self):
        return self._ok

    def busy(self):
        """True while anything is queued or sounding. Lets a caller step through
        lines one at a time (read-along) instead of dumping them all at once."""
        return self._speaking or not self.q.empty()

    def cancel(self):
        """Stop the current utterance and drop the backlog, without muting --
        pause/stop for read-along, which must be able to resume."""
        self._drain()
        self._cancel = True

    def set_muted(self, muted):
        self._muted = bool(muted)
        if muted:                       # flush pending speech immediately
            self._drain()

    def _drain(self):
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass

    def say(self, text):
        """Queue text to be spoken. Drops silently if muted, unavailable, or the
        queue is full (prevents unbounded backlog on busy feeds)."""
        if not self._ok or self._muted or not text:
            return
        try:
            self.q.put_nowait(text)
        except queue.Full:
            pass

    def run(self):
        try:
            import sounddevice as sd
        except Exception as e:
            self._ok = False
            if self.out:
                self.out.status(f"TTS unavailable (no audio output): {e}")
            return
        # Build the synth engine ON this thread (required for SAPI/COM).
        try:
            self._synth = _make_tts_synth(self._engine, self._voice_id)
            self._voice_id = self._synth.voice_id
            self._sr = self._synth.sample_rate
        except Exception as e:
            self._ok = False
            if self.out:
                self.out.status(f"TTS: failed to load voice: {e}")
            return
        while not self.stop_evt.is_set():
            try:
                text = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            if self._muted:
                continue
            try:
                audio = self._synth.synthesize(text)
                if audio is None or len(audio) == 0:
                    continue
                # Re-read the rate: some backends (WinRT) learn it per utterance.
                self._sr = getattr(self._synth, "sample_rate", self._sr)
                if self.on_start:
                    try:
                        self.on_start(text)
                    except Exception:
                        pass
                self._cancel = False        # only cancels from here on
                self._speaking = True
                sd.play(audio, self._sr)
                # Wait for playback, but bail out promptly on stop/mute/cancel.
                while sd.get_stream().active and not self.stop_evt.is_set() \
                        and not self._muted and not self._cancel:
                    time.sleep(0.05)
                if self._muted or self.stop_evt.is_set() or self._cancel:
                    sd.stop()
            except Exception as e:
                if self.out:
                    self.out.status(f"TTS error: {e}")
            finally:
                self._speaking = False
                if self.on_end:
                    try:
                        self.on_end(text)
                    except Exception:
                        pass

    def close(self):
        self.stop_evt.set()
        self._drain()


# --------------------------------------------------------------------------
# Adaptive voice/energy gate -> carves the stream into transmissions
# For continuous audio (TV/streaming) there is rarely a silence gap to end a
# segment, so it would grow until max_segment_sec -> big latency. These defaults
# flush far sooner. Applied automatically to pcaudio streams.
PCAUDIO_VAD_DEFAULTS = {
    "max_segment_sec": 6.0,        # flush at least every 6s even with no silence
    "silence_hangover_sec": 0.5,   # end a bit sooner on the pauses that do occur
    # Continuous audio (TV/streaming) has no radio-style silence gaps and the
    # energy gate drops too much of it. In continuous mode we DON'T gate -- we
    # capture everything in fixed chunks and let Whisper's no_speech filter sort
    # speech from music/silence. This is the right model for TV/app audio.
    "continuous": True,
    "chunk_sec": 5.0,              # fixed chunk length in continuous mode
}


def effective_vad(base_vad, stream):
    """Merge VAD config for a stream: base config, then pcaudio fast-flush
    defaults (if applicable), then any per-stream 'vad' override. Lower
    max_segment_sec = lower latency for continuous audio."""
    cfg = dict(base_vad or {})
    if stream.get("type") in ("pcaudio", "app"):
        cfg.update(PCAUDIO_VAD_DEFAULTS)      # fast-flush for continuous audio
    cfg.update(stream.get("vad", {}) or {})   # explicit per-stream override wins
    return cfg


# --------------------------------------------------------------------------
class SpeechGate:
    """
    Tracks a running background-noise floor and triggers a 'transmission' when
    energy rises clearly above it. Emits buffered audio when the transmission
    ends (silence hangover) or hits the max length. Pre-roll keeps the onset.
    """
    def __init__(self, vad_cfg):
        self.trigger_ratio = vad_cfg.get("trigger_ratio", 3.0)
        self.abs_floor = vad_cfg.get("abs_min_rms", 0.004)
        self.hangover_sec = vad_cfg.get("silence_hangover_sec", 0.8)
        self.min_speech_sec = vad_cfg.get("min_speech_sec", 0.4)
        self.max_segment_sec = vad_cfg.get("max_segment_sec", 25.0)
        self.preroll_sec = vad_cfg.get("preroll_sec", 0.3)

        # Continuous mode: emit fixed chunks, no energy gating (for TV/app audio).
        self.continuous = bool(vad_cfg.get("continuous", False))
        self.chunk_sec = vad_cfg.get("chunk_sec", 5.0)

        self.noise_floor = self.abs_floor
        self.in_speech = False
        self.silence_run = 0.0
        self.speech_len = 0.0
        self.buf = []                                  # frames in current segment
        self.preroll = []                              # recent pre-speech frames
        self.preroll_max = int(self.preroll_sec * 1000 / FRAME_MS)

    @staticmethod
    def _rms(frame):
        if frame.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(frame.astype(np.float32) ** 2)))

    def push(self, frame):
        """Feed one float32 frame. Returns a completed segment (np.array) or None."""
        # Continuous mode: accumulate and emit a fixed-length chunk -- no gating,
        # so nothing is dropped. Whisper's no_speech filter handles silence/music.
        if self.continuous:
            self.buf.append(frame)
            self.speech_len += FRAME_MS / 1000.0
            if self.speech_len >= self.chunk_sec:
                seg = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
                self.buf = []
                self.speech_len = 0.0
                return seg
            return None

        rms = self._rms(frame)
        threshold = max(self.abs_floor, self.noise_floor * self.trigger_ratio)
        voiced = rms > threshold

        if not self.in_speech:
            # Adapt the noise floor only while idle (slow EMA).
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * rms
            self.preroll.append(frame)
            if len(self.preroll) > self.preroll_max:
                self.preroll.pop(0)
            if voiced:
                self.in_speech = True
                self.buf = list(self.preroll)
                self.preroll = []
                self.silence_run = 0.0
                self.speech_len = 0.0
            return None

        # In speech: accumulate.
        self.buf.append(frame)
        self.speech_len += FRAME_MS / 1000.0
        self.silence_run = 0.0 if voiced else self.silence_run + FRAME_MS / 1000.0

        ended = self.silence_run >= self.hangover_sec
        too_long = self.speech_len >= self.max_segment_sec
        if ended or too_long:
            seg = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
            had_enough = self.speech_len >= self.min_speech_sec
            self.in_speech = False
            self.buf = []
            self.silence_run = 0.0
            self.speech_len = 0.0
            return seg if had_enough else None
        return None


# --------------------------------------------------------------------------
# Stream worker: ffmpeg URL -> PCM -> SpeechGate -> transcription queue
# --------------------------------------------------------------------------
class StreamWorker(threading.Thread):
    def __init__(self, stream, ffmpeg, vad_cfg, jobq, out, stop_evt,
                 auth_header=None, player=None):
        super().__init__(daemon=True, name=stream["name"])
        self.name_ = stream["name"]
        self.url = normalize_url(stream["url"], stream.get("provider"))
        self.color = stream.get("color", "white")
        self.ffmpeg = ffmpeg
        # Send HTTP Basic auth only to Broadcastify's audio host, never to
        # arbitrary third-party streams (avoids leaking creds off-site).
        self.auth_header = auth_header if "audio.broadcastify.com" in self.url else None
        self.vad_cfg = effective_vad(vad_cfg, stream)
        self.jobq = jobq
        self.out = out
        self.stop_evt = stop_evt        # shared global stop
        self.own_stop = threading.Event()  # per-worker stop (dynamic removal)
        self.player = player
        self._proc = None

    def _stopping(self):
        return self.stop_evt.is_set() or self.own_stop.is_set()

    def stop(self):
        """Signal just this worker to stop and kill its ffmpeg promptly."""
        self.own_stop.set()
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass

    def _spawn_ffmpeg(self):
        cmd = [
            self.ffmpeg,
            "-nostdin", "-loglevel", "error",
            "-user_agent", "Mozilla/5.0",
            "-reconnect", "1", "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
        ]
        if self.auth_header:
            # Pass credentials as a real Authorization header rather than in the
            # URL: handles special characters in the password and keeps it out
            # of the visible -i argument.
            cmd += ["-headers", f"Authorization: Basic {self.auth_header}\r\n"]
        cmd += [
            "-i", self.url,
            "-f", "s16le", "-acodec", "pcm_s16le",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-",
        ]
        return subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=FRAME_BYTES * 8, **_no_window_kwargs(),
        )

    def run(self):
        backoff = 1.0
        while not self._stopping():
            self.out.status(f"[{self.name_}] connecting to stream...")
            try:
                self._proc = proc = self._spawn_ffmpeg()
            except Exception as e:
                self.out.status(f"[{self.name_}] ffmpeg launch failed: {e}")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue

            gate = SpeechGate(self.vad_cfg)
            pending = b""
            got_audio = False
            try:
                while not self._stopping():
                    chunk = proc.stdout.read(FRAME_BYTES * 4)
                    if not chunk:
                        break  # stream ended / dropped
                    got_audio = True
                    backoff = 1.0
                    # Feed speakers (player emits only if this is the selected source).
                    if self.player is not None:
                        self.player.feed(self.name_, chunk)
                    pending += chunk
                    while len(pending) >= FRAME_BYTES:
                        raw, pending = pending[:FRAME_BYTES], pending[FRAME_BYTES:]
                        frame = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                        seg = gate.push(frame)
                        if seg is not None:
                            self.jobq.put((self.name_, self.color, seg, time.time()))
            finally:
                try:
                    proc.kill()
                except Exception:
                    pass
                self._proc = None

            if self._stopping():
                break
            wait = 1.0 if got_audio else backoff
            self.out.status(f"[{self.name_}] stream dropped; reconnecting in {wait:.0f}s")
            time.sleep(wait)
            if not got_audio:
                backoff = min(backoff * 2, 30)


# --------------------------------------------------------------------------
# PC-audio capture helpers + worker.
#
# Captures from a Windows input device (e.g. "Stereo Mix", which mirrors
# everything playing on the PC) at its native rate, downmixes to mono, and
# resamples to 16 kHz for Whisper. Same SpeechGate/jobq path as StreamWorker.
# --------------------------------------------------------------------------
def list_input_devices():
    """Return [(index, name, default_samplerate)] for capture-capable devices.
    Empty list if sounddevice/PortAudio is unavailable.

    Excludes WDM-KS host-API devices: they don't support the blocking stream API
    we use (PortAudio error -9999 'Blocking API not supported yet'). The same
    physical device is still listed under MME/DirectSound/WASAPI, which work."""
    try:
        import sounddevice as sd
    except Exception:
        return []
    try:
        hostapis = sd.query_hostapis()
    except Exception:
        hostapis = []
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) <= 0:
            continue
        ha = d.get("hostapi")
        ha_name = hostapis[ha]["name"] if (isinstance(ha, int) and ha < len(hostapis)) else ""
        if "WDM-KS" in ha_name or "Kernel Streaming" in ha_name:
            continue   # blocking API unsupported -> would fail at capture time
        devices.append((i, d["name"], int(d.get("default_samplerate") or 44100)))
    return devices


# Names that indicate a device captures the PC's OUTPUT (what you hear), as
# opposed to a microphone capturing the room.
_LOOPBACK_KEYWORDS = ("stereo mix", "what u hear", "what you hear", "loopback",
                      "wave out", "speakers", "voicemeeter out", "cable output")
# Names that indicate a physical microphone -- never auto-pick these for PC audio.
_MIC_KEYWORDS = ("microphone", "mic ", "webcam", "headset", "line in", "mic input")


def is_loopback_name(name):
    """True if a device name looks like an output-capture (loopback) device."""
    low = (name or "").lower()
    if any(k in low for k in _MIC_KEYWORDS):
        return False
    return any(k in low for k in _LOOPBACK_KEYWORDS)


def verify_device_streamable(device_index, timeout=0.3):
    """Actually open a brief input stream to confirm the device can be captured
    (some host APIs list devices that fail at stream time, e.g. WDM-KS -9999).
    Returns True if a stream opens and reads, else False."""
    try:
        import sounddevice as sd
        info = sd.query_devices(device_index)
        sr = int(info.get("default_samplerate") or 48000)
        ch = max(1, int(info.get("max_input_channels", 1)))
        with sd.InputStream(device=device_index, samplerate=sr, channels=ch,
                            dtype="float32", blocksize=int(sr * 0.05)) as st:
            st.read(int(sr * 0.05))
        return True
    except Exception:
        return False


def find_loopback_device():
    """Best-effort index of an output-capture device ('Stereo Mix' / loopback)
    that actually streams. Prefers a verified-streamable one; falls back to the
    first by name. Returns None if there are no loopback devices at all."""
    candidates = [idx for idx, name, _sr in list_input_devices() if is_loopback_name(name)]
    for idx in candidates:
        if verify_device_streamable(idx):
            return idx
    return candidates[0] if candidates else None


def probe_device_level(device_index, seconds=0.6):
    """Record a brief sample from one input device and return its RMS level
    (0.0 if it can't be opened or is silent). Lets the GUI show which device is
    actually receiving audio right now, so the user picks the right one."""
    try:
        import sounddevice as sd
    except Exception:
        return 0.0
    try:
        info = sd.query_devices(device_index)
        sr = int(info.get("default_samplerate") or 48000)
        ch = max(1, int(info.get("max_input_channels", 1)))
        rec = sd.rec(int(seconds * sr), samplerate=sr, channels=ch,
                     dtype="float32", device=device_index)
        sd.wait()
        if rec.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(rec.astype(np.float32) ** 2)))
    except Exception:
        return 0.0


def probe_device_levels(seconds=0.6):
    """Probe every input device; return [(index, name, rms_level, is_loopback)]
    sorted loudest-first. The is_loopback flag lets callers prefer output-capture
    devices over microphones (a mic hears the room and is misleadingly 'loud')."""
    results = []
    for idx, name, _sr in list_input_devices():
        results.append((idx, name, probe_device_level(idx, seconds),
                        is_loopback_name(name)))
    results.sort(key=lambda r: r[2], reverse=True)
    return results


def best_loopback_by_signal(levels, verify=True):
    """Given probe_device_levels() output, return the index of the loudest
    LOOPBACK device with real signal that can actually be streamed, or None.
    Never a microphone. With verify=True, skips devices that fail to open."""
    loop = sorted((r for r in levels if r[3] and r[2] > 0.0005),
                  key=lambda r: r[2], reverse=True)
    for r in loop:
        if not verify or verify_device_streamable(r[0]):
            return r[0]
    return None


def _resample_to_16k(mono, src_rate):
    """Linear resample a float32 mono array from src_rate to 16 kHz. Cheap and
    dependency-free; speech transcription doesn't need a fancy anti-alias filter."""
    if src_rate == SAMPLE_RATE or mono.size == 0:
        return mono.astype(np.float32)
    n_out = int(round(mono.size * SAMPLE_RATE / src_rate))
    if n_out <= 0:
        return np.zeros(0, np.float32)
    x_old = np.linspace(0.0, 1.0, num=mono.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, mono).astype(np.float32)


class DeviceWorker(threading.Thread):
    """Capture a PC audio input device -> mono 16k -> SpeechGate -> jobq."""
    def __init__(self, stream, vad_cfg, jobq, out, stop_evt):
        super().__init__(daemon=True, name=stream["name"])
        self.name_ = stream["name"]
        self.color = stream.get("color", "white")
        self.device = stream.get("device")          # input device index (int)
        self.vad_cfg = effective_vad(vad_cfg, stream)
        self.jobq = jobq
        self.out = out
        self.stop_evt = stop_evt
        self.own_stop = threading.Event()

    def _stopping(self):
        return self.stop_evt.is_set() or self.own_stop.is_set()

    def stop(self):
        self.own_stop.set()

    def run(self):
        try:
            import sounddevice as sd
        except Exception as e:
            self.out.status(f"[{self.name_}] audio capture unavailable: {e}")
            return

        try:
            info = sd.query_devices(self.device)
        except Exception as e:
            self.out.status(f"[{self.name_}] bad capture device: {e}")
            return
        src_rate = int(info.get("default_samplerate") or 48000)
        in_ch = max(1, int(info.get("max_input_channels", 1)))
        gate = SpeechGate(self.vad_cfg)
        blocksize = int(src_rate * FRAME_MS / 1000)  # ~one VAD frame per callback

        self.out.status(f"[{self.name_}] capturing PC audio "
                        f"('{info['name']}' @ {src_rate}Hz)...")
        backoff = 1.0
        while not self._stopping():
            try:
                # Blocking read loop (not a callback) so all gate/queue work
                # stays on this thread, matching StreamWorker's model.
                with sd.InputStream(device=self.device, samplerate=src_rate,
                                    channels=in_ch, dtype="float32",
                                    blocksize=blocksize) as stream:
                    while not self._stopping():
                        data, _overflowed = stream.read(blocksize)
                        if data.size == 0:
                            continue
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        mono = _resample_to_16k(np.asarray(mono, dtype=np.float32), src_rate)
                        for i in range(0, len(mono) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                            seg = gate.push(mono[i:i + FRAME_SAMPLES])
                            if seg is not None:
                                self.jobq.put((self.name_, self.color, seg, time.time()))
                backoff = 1.0
            except Exception as e:
                if self._stopping():
                    break
                self.out.status(f"[{self.name_}] capture error: {e}; retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)


# --------------------------------------------------------------------------
# Output-device loopback capture via the `soundcard` library.
#
# Unlike "Stereo Mix" (one Realtek-only input that proved unreliable here),
# soundcard can loopback-capture ANY output device by name -- Realtek speakers,
# an external DAC, HDMI, etc. This is the preferred PC-audio path. To capture a
# specific app, route that app to a given output device in Windows, then select
# that device here.
# --------------------------------------------------------------------------
def soundcard_available():
    try:
        import soundcard  # noqa: F401
        return True
    except Exception:
        return False


_OUTPUT_DEVICES = None          # cached [(name, is_default)]; see warm_audio_devices


def warm_audio_devices():
    """Enumerate output devices ONCE, before any Tk window exists.

    On Windows-on-ARM64, enumerating through soundcard AFTER Tk has started
    corrupts the heap and kills the process (0xC0000374) -- Tk initialises OLE,
    and soundcard's COM use conflicts when it initialises second. Doing it first
    and serving a cache afterwards avoids the collision entirely. It's also just
    faster: the add/edit dialog no longer re-queries WASAPI every time it opens.

    Call this before creating the root window. Safe to call more than once."""
    return list_output_devices(force=True)


def list_output_devices(force=False):
    """Return [(name, is_default)] of output devices that can be loopback-captured
    via soundcard. Empty if soundcard is unavailable. Names are de-duplicated.

    Served from the cache once warmed (see warm_audio_devices) -- so a device
    plugged in mid-session won't appear until restart, which is the price of not
    crashing the app on ARM."""
    global _OUTPUT_DEVICES
    if _OUTPUT_DEVICES is not None and not force:
        return list(_OUTPUT_DEVICES)
    try:
        import soundcard as sc
    except Exception:
        _OUTPUT_DEVICES = []
        return []
    try:
        default_name = sc.default_speaker().name
    except Exception:
        default_name = None
    seen, out = set(), []
    for sp in sc.all_speakers():
        if sp.name in seen:
            continue
        seen.add(sp.name)
        out.append((sp.name, sp.name == default_name))
    return out


def _sc_loopback_mic(name):
    """Get the soundcard loopback 'microphone' that captures output device `name`."""
    import soundcard as sc
    return sc.get_microphone(name, include_loopback=True)


def probe_output_level(name, seconds=0.5):
    """RMS level currently coming out of output device `name` (0.0 on failure)."""
    try:
        import soundcard as sc, numpy as _np  # noqa
        mic = _sc_loopback_mic(name)
        with mic.recorder(samplerate=SAMPLE_RATE, channels=1) as r:
            data = r.record(numframes=int(SAMPLE_RATE * seconds))
        if data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.asarray(data, dtype=np.float32) ** 2)))
    except Exception:
        return 0.0


def probe_output_levels(seconds=0.5):
    """Probe every output device's current loopback level; loudest first.
    Returns [(name, rms, is_default)]."""
    results = []
    for name, is_def in list_output_devices():
        results.append((name, probe_output_level(name, seconds), is_def))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


class LoopbackWorker(threading.Thread):
    """Capture an OUTPUT device via soundcard loopback -> mono 16k -> gate -> jobq.
    Used for pcaudio streams that specify an output device by name."""
    def __init__(self, stream, vad_cfg, jobq, out, stop_evt):
        super().__init__(daemon=True, name=stream["name"])
        self.name_ = stream["name"]
        self.color = stream.get("color", "white")
        self.out_device = stream.get("output_device")   # output device NAME
        self.vad_cfg = effective_vad(vad_cfg, stream)
        self.jobq = jobq
        self.out = out
        self.stop_evt = stop_evt
        self.own_stop = threading.Event()

    def _stopping(self):
        return self.stop_evt.is_set() or self.own_stop.is_set()

    def stop(self):
        self.own_stop.set()

    def run(self):
        try:
            import soundcard as sc  # noqa
        except Exception as e:
            self.out.status(f"[{self.name_}] soundcard unavailable: {e}")
            return
        backoff = 1.0
        chunk_frames = int(SAMPLE_RATE * 0.1)   # 100ms reads at 16k
        self.out.status(f"[{self.name_}] capturing output '{self.out_device}'...")
        while not self._stopping():
            gate = SpeechGate(self.vad_cfg)
            try:
                mic = _sc_loopback_mic(self.out_device)
                # soundcard resamples to the requested samplerate for us.
                with mic.recorder(samplerate=SAMPLE_RATE, channels=1,
                                  blocksize=chunk_frames) as rec:
                    while not self._stopping():
                        data = rec.record(numframes=chunk_frames)
                        if data is None or len(data) == 0:
                            continue
                        mono = data[:, 0] if getattr(data, "ndim", 1) > 1 else data
                        mono = np.asarray(mono, dtype=np.float32)
                        for i in range(0, len(mono) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                            seg = gate.push(mono[i:i + FRAME_SAMPLES])
                            if seg is not None:
                                self.jobq.put((self.name_, self.color, seg, time.time()))
                backoff = 1.0
            except Exception as e:
                if self._stopping():
                    break
                self.out.status(f"[{self.name_}] capture error: {e}; retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 15)


# --------------------------------------------------------------------------
# Per-application capture (WASAPI process loopback via the `proctap` library).
#
# Captures audio from ONE process (by PID) and its children -- so you can
# transcribe a specific app (e.g. a media player) regardless of which output
# device it uses. Caveat: apps that share one process tree (e.g. all Chrome
# tabs) can't be separated from each other.
# --------------------------------------------------------------------------
def proctap_available():
    """True only if per-app capture can ACTUALLY run. proc-tap ships a pure-python
    (py3-none-any) wheel whose compiled `_native` extension has no Windows-ARM64
    build, so the package imports fine there while every capture raises. Require
    the native extension too, otherwise the GUI would offer an 'application' source
    that always fails.

    This deliberately answers from DISK and never imports proctap. On ARM64,
    importing it and then enumerating speakers through soundcard corrupts the
    heap and kills the process (0xC0000374) -- and the two are probed together
    all over: the add-feed dialog offers both sources, and importing a feed list
    can carry both kinds. find_spec on a top-level name doesn't execute it;
    find_spec("proctap._native") WOULD, because locating a submodule imports its
    parent package. So we look for the .pyd beside __init__.py ourselves."""
    try:
        import importlib.util
        spec = importlib.util.find_spec("proctap")     # does not import it
        if spec is None or not spec.origin:
            return False
        pkg_dir = os.path.dirname(spec.origin)
        return any(f.startswith("_native") and f.endswith((".pyd", ".so"))
                   for f in os.listdir(pkg_dir))
    except Exception:
        return False


def list_audio_apps():
    """Return [(pid, exe_name, is_active)] for processes that currently have an
    audio session (i.e. can produce sound). is_active=True means it's playing
    right now. Empty list if pycaw is unavailable. De-duplicated by pid."""
    try:
        from pycaw.pycaw import AudioUtilities
    except Exception:
        return []
    apps, seen = [], set()
    try:
        sessions = AudioUtilities.GetAllSessions()
    except Exception:
        return []
    for s in sessions:
        proc = getattr(s, "Process", None)
        if proc is None:
            continue
        try:
            pid = proc.pid
            if pid in seen:
                continue
            seen.add(pid)
            name = proc.name()
            active = getattr(s, "State", 0) == 1   # AudioSessionStateActive
            apps.append((pid, name, active))
        except Exception:
            continue
    # Active (currently-playing) apps first, then by name.
    apps.sort(key=lambda a: (not a[2], a[1].lower()))
    return apps


PROCTAP_SAMPLE_RATE = 48000   # proctap's Windows backend output rate


class ProcessLoopbackWorker(threading.Thread):
    """Capture one process's audio (by PID) via proctap -> mono 16k -> gate -> jobq."""
    def __init__(self, stream, vad_cfg, jobq, out, stop_evt):
        super().__init__(daemon=True, name=stream["name"])
        self.name_ = stream["name"]
        self.color = stream.get("color", "white")
        self.pid = stream.get("pid")
        self.app_name = stream.get("app_name", "")
        self.vad_cfg = effective_vad(vad_cfg, stream)
        self.jobq = jobq
        self.out = out
        self.stop_evt = stop_evt
        self.own_stop = threading.Event()
        self._buf = bytearray()
        self._buf_lock = threading.Lock()

    def _stopping(self):
        return self.stop_evt.is_set() or self.own_stop.is_set()

    def stop(self):
        self.own_stop.set()

    def _on_data(self, data, _ts):
        # proctap delivers float32 stereo @ 48k. Buffer raw bytes; the run loop
        # downmixes + resamples on its own thread.
        with self._buf_lock:
            self._buf.extend(data)

    def run(self):
        try:
            import proctap
        except Exception as e:
            self.out.status(f"[{self.name_}] per-app capture unavailable: {e}")
            return
        if not self.pid:
            self.out.status(f"[{self.name_}] no process selected.")
            return

        gate = SpeechGate(self.vad_cfg)
        self.out.status(f"[{self.name_}] capturing app '{self.app_name}' (pid {self.pid})...")
        cap = None
        try:
            cap = proctap.ProcessAudioCapture(pid=int(self.pid), on_data=self._on_data)
            cap.start()
            # stereo float32 @ 48k -> bytes per 48k frame = 2ch * 4 bytes
            bytes_per_frame = 2 * 4
            while not self._stopping():
                with self._buf_lock:
                    chunk = bytes(self._buf)
                    self._buf.clear()
                if not chunk:
                    time.sleep(0.03)
                    continue
                stereo = np.frombuffer(chunk, dtype=np.float32)
                # Trim to whole frames, downmix to mono, resample 48k -> 16k.
                n = (len(stereo) // 2) * 2
                if n == 0:
                    continue
                stereo = stereo[:n].reshape(-1, 2)
                mono = stereo.mean(axis=1)
                mono16 = _resample_to_16k(mono, PROCTAP_SAMPLE_RATE)
                for i in range(0, len(mono16) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                    seg = gate.push(mono16[i:i + FRAME_SAMPLES])
                    if seg is not None:
                        self.jobq.put((self.name_, self.color, seg, time.time()))
        except Exception as e:
            if not self._stopping():
                self.out.status(f"[{self.name_}] app capture error: {e}")
        finally:
            try:
                if cap is not None:
                    cap.stop(); cap.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
# Anti-hallucination helpers.
#
# Whisper "fills in" common training-data phrases on silence/non-speech (e.g.
# "Thank you", "Thanks for watching", "Please subscribe"). It can also loop,
# emitting the same phrase many times. We drop pure-hallucination segments and
# collapse repeated phrases.
# --------------------------------------------------------------------------
# Phrases Whisper commonly emits over silence -- dropped if the segment is ONLY
# this (and the model wasn't confident it was speech).
_HALLUCINATION_PHRASES = {
    "thank you", "thank you.", "thanks for watching", "thanks for watching.",
    "thank you for watching", "thank you for watching.", "please subscribe",
    "please subscribe.", "subscribe", "you", "you.", "bye", "bye.",
    "thanks for watching!", "thank you very much", "thank you very much.",
    ".", "..", "...",
}


def _is_hallucination(text, no_speech_prob):
    """True if `text` is just a known silence-hallucination phrase (optionally
    repeated) and the model wasn't confident this was speech."""
    low = text.strip().lower()
    # Collapse internal repeats first ("thank you. thank you." -> "thank you.")
    collapsed = _collapse_repeats(low)
    if collapsed in _HALLUCINATION_PHRASES and no_speech_prob > 0.35:
        return True
    return False


def _collapse_repeats(text):
    """Collapse immediate repeated phrases: 'Thank you. Thank you. Thank you.'
    -> 'Thank you.' Works on sentence-ish units split by . ? ! and on repeated
    single words. Conservative: only collapses 3+ identical consecutive units."""
    if not text:
        return text
    # Split into sentence-ish chunks, keeping the delimiter.
    import re as _re
    units = _re.findall(r"[^.?!]+[.?!]?", text)
    units = [u.strip() for u in units if u.strip()]
    out, i = [], 0
    while i < len(units):
        j = i
        while j < len(units) and units[j].lower() == units[i].lower():
            j += 1
        run = j - i
        # Keep one copy if a phrase repeats 3+ times (clear loop); else keep all.
        out.append(units[i] if run >= 3 else " ".join(units[i:j]))
        i = j
    result = " ".join(out)
    # Also squash repeated single words ("you you you you" -> "you").
    result = _re.sub(r"\b(\w+)(\s+\1\b){2,}", r"\1", result, flags=_re.I)
    return result.strip()


# --------------------------------------------------------------------------
# Transcription backends.
#
# The rest of the app talks to ONE object with a faster-whisper-shaped
# `.transcribe(audio, **kw) -> (segments, info)` API, where each segment exposes
# `.text`, `.no_speech_prob`, and `.avg_logprob`. Two implementations:
#   * faster-whisper / ctranslate2 -- default on x64 (GPU or CPU). Its
#     WhisperModel already has exactly this shape, so it's used directly.
#   * whisper.cpp via pywhispercpp  -- default on native ARM64, where ctranslate2
#     has no wheel. Runs on CPU/NPU. `WhisperCppBackend` adapts it to the shape.
# `Engine._make_whisper_model()` picks one; `Transcriber` and the anti-
# hallucination filters are backend-agnostic because the shape is identical.
# --------------------------------------------------------------------------
def select_backend(cfg):
    """'ct2' (faster-whisper) or 'whispercpp'. Explicit cfg['engine'] wins;
    otherwise 'whispercpp' on a native-ARM64 interpreter (no ctranslate2 wheel),
    'ct2' elsewhere. Keys off the real process arch, NOT platform.machine()
    (see interpreter_is_arm64)."""
    engine = str(cfg.get("engine") or "").strip().lower()
    if engine in ("ct2", "ctranslate2", "faster-whisper", "faster_whisper"):
        return "ct2"
    if engine in ("whispercpp", "whisper.cpp", "whisper_cpp", "pywhispercpp"):
        return "whispercpp"
    return "whispercpp" if interpreter_is_arm64() else "ct2"


# faster-whisper model id -> nearest whisper.cpp (GGML) model name. whisper.cpp
# ships its own GGML files (see pywhispercpp AVAILABLE_MODELS); faster-whisper's
# "distil-*" models have no GGML build, so they map to the closest standard one.
_WHISPERCPP_MODEL_MAP = {
    "large-v3": "large-v3", "large-v2": "large-v2", "large-v1": "large-v1",
    "large-v3-turbo": "large-v3-turbo",
    "medium": "medium", "medium.en": "medium.en",
    "small": "small", "small.en": "small.en",
    "base": "base", "base.en": "base.en",
    "tiny": "tiny", "tiny.en": "tiny.en",
    "distil-large-v3": "large-v3-turbo", "distil-large-v2": "large-v2",
    "distil-medium.en": "medium.en", "distil-small.en": "small.en",
}
# CPU/NPU inference is far slower than the x64 GPU path, so ARM configs should
# choose a small/quantized model; this is the fallback when a name can't be mapped.
ARM_DEFAULT_MODEL = "small.en-q5_1"

# Approximate GGML download sizes (MB), for a friendly first-run download status.
# Rough figures -- only used to show "~N MB" while the model downloads.
_WHISPERCPP_MODEL_MB = {
    "tiny": 75, "tiny.en": 75, "tiny-q5_1": 31, "tiny.en-q5_1": 31,
    "base": 148, "base.en": 148, "base-q5_1": 57, "base.en-q5_1": 57,
    "small": 488, "small.en": 488, "small-q5_1": 181, "small.en-q5_1": 181,
    "medium": 1530, "medium.en": 1530,
    "large-v3": 3100, "large-v3-turbo": 1600, "large-v2": 3100,
}


def whispercpp_model_name(name):
    """Resolve an app model id to a valid whisper.cpp GGML model name."""
    try:
        from pywhispercpp import constants as _c
        avail = set(getattr(_c, "AVAILABLE_MODELS", []))
    except Exception:
        avail = set()
    if not avail:                       # can't validate -> best-effort passthrough
        return _WHISPERCPP_MODEL_MAP.get(name, name)
    if name in avail:
        return name
    mapped = _WHISPERCPP_MODEL_MAP.get(name)
    if mapped in avail:
        return mapped
    return ARM_DEFAULT_MODEL if ARM_DEFAULT_MODEL in avail else "small.en"


class _WCSegment:
    """A faster-whisper-shaped segment (only the fields Transcriber reads)."""
    __slots__ = ("text", "no_speech_prob", "avg_logprob")

    def __init__(self, text, no_speech_prob, avg_logprob):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


def _map_whispercpp_segments(segs):
    """Adapt pywhispercpp Segments -> faster-whisper-shaped segments.

    pywhispercpp gives one confidence number per segment: `probability`, the
    geometric mean of token probabilities in [0, 1] (NaN if not computed). Whisper
    proper exposes two independent numbers the filters use -- avg_logprob and
    no_speech_prob -- which whisper.cpp doesn't surface per segment. Synthesize
    both from `probability` p:
        avg_logprob    = log(p)   -> the min_avg_logprob gate drops low-confidence
                                     garbage (p < e^-1 ~= 0.37 with the default).
        no_speech_prob = 1 - p    -> keeps the phrase anti-hallucination filter
                                     (needs no_speech_prob > 0.35) meaningful; only
                                     marginally stricter than the logprob gate.
    A NaN probability -> neutral scores that pass both gates, so nothing is dropped
    merely for lacking a confidence number."""
    out = []
    for s in segs:
        try:
            p = float(getattr(s, "probability", float("nan")))
        except (TypeError, ValueError):
            p = float("nan")
        if p != p:                      # NaN
            no_speech, avg_logprob = 0.0, 0.0
        else:
            p = min(max(p, 0.0), 1.0)
            no_speech = 1.0 - p
            avg_logprob = math.log(p) if p > 0.0 else -10.0
        out.append(_WCSegment(getattr(s, "text", ""), no_speech, avg_logprob))
    return out


class WhisperCppBackend:
    """whisper.cpp (via pywhispercpp) with a faster-whisper-shaped transcribe().

    Default backend on native Windows-on-ARM, where ctranslate2 (and thus
    faster-whisper) has no wheel. Runs on CPU/NPU. GGML model files are downloaded
    on first use into a writable models dir. `model=` lets tests inject a fake."""
    def __init__(self, model_name, cfg, status_cb=None, model=None):
        self.cfg = cfg
        self._status = status_cb
        self._n_threads = int(cfg.get("n_threads") or max(1, (os.cpu_count() or 4)))
        self._lang = cfg.get("language", "en") or "en"
        if model is not None:                 # injected (tests) -- skip real load
            self._model = model
            return
        from pywhispercpp.model import Model
        wname = whispercpp_model_name(model_name)
        models_dir = cfg.get("whispercpp_models_dir") or os.path.join(
            DATA_DIR, "whispercpp_models")
        os.makedirs(models_dir, exist_ok=True)

        # First use downloads the GGML model (up to hundreds of MB) into models_dir.
        # In a packaged app there's no console, and Model() blocks with no visible
        # feedback -- so it just looks frozen / "not transcribing". Report progress
        # to the GUI status line: pywhispercpp writes the file as it downloads, so a
        # watcher thread reports how many MB have landed while Model() fetches it.
        model_file = os.path.join(models_dir, f"ggml-{wname}.bin")
        downloading = not os.path.exists(model_file)
        watch_stop = threading.Event()
        if downloading and status_cb:
            approx = _WHISPERCPP_MODEL_MB.get(wname)
            status_cb(f"First run: downloading the '{wname}' speech model"
                      + (f" (~{approx} MB)" if approx else "")
                      + " — one time. Transcription starts when it finishes…")

            def _watch():
                while not watch_stop.wait(2.0):
                    try:
                        mb = os.path.getsize(model_file) / (1 << 20)
                    except OSError:
                        mb = 0
                    status_cb(f"Downloading '{wname}' model… {mb:.0f}"
                              + (f" / ~{approx} MB" if approx else " MB"))
            threading.Thread(target=_watch, daemon=True, name="wcpp-dl").start()
        elif status_cb:
            status_cb(f"Loading whisper.cpp model '{wname}' "
                      f"({self._n_threads} threads)...")

        try:
            self._model = Model(
                model=wname, models_dir=models_dir,
                redirect_whispercpp_logs_to=False,
                n_threads=self._n_threads, print_progress=False,
                print_realtime=False,
            )
        finally:
            watch_stop.set()
        if downloading and status_cb:
            status_cb(f"Model '{wname}' downloaded ({self._n_threads} threads).")

    def transcribe(self, audio, language=None, initial_prompt=None,
                   no_speech_threshold=0.6, log_prob_threshold=-1.0,
                   compression_ratio_threshold=2.4, **_ignored):
        """Mirror faster-whisper's WhisperModel.transcribe signature + return
        shape. Unmapped kwargs (beam_size, vad_filter, temperature,
        condition_on_previous_text, no_repeat_ngram_size, ...) are accepted and
        ignored -- whisper.cpp handles the equivalents internally or upstream."""
        a = np.ascontiguousarray(audio, dtype=np.float32)
        segs = self._model.transcribe(
            a,
            language=language or self._lang,
            initial_prompt=initial_prompt or "",
            no_context=True,                    # == condition_on_previous_text=False
            translate=False,
            print_progress=False,
            single_segment=False,
            no_speech_thold=float(no_speech_threshold),
            logprob_thold=float(log_prob_threshold),
            entropy_thold=float(compression_ratio_threshold),
            temperature=0.0,
            extract_probability=True,
        )
        info = {"language": language or self._lang, "backend": "whispercpp"}
        return _map_whispercpp_segments(segs), info


# --------------------------------------------------------------------------
# Transcription worker: single shared model serving all streams
# --------------------------------------------------------------------------
class Transcriber(threading.Thread):
    def __init__(self, model, cfg, jobq, out, stop_evt, clips=None,
                 should_record=None, prompt_for=None):
        super().__init__(daemon=True, name="transcriber")
        self.model = model
        self.cfg = cfg
        self.jobq = jobq
        self.out = out
        self.stop_evt = stop_evt
        self.max_no_speech = cfg.get("filters", {}).get("max_no_speech_prob", 0.6)
        self.min_logprob = cfg.get("filters", {}).get("min_avg_logprob", -1.0)
        self.tts_hook = None    # optional callable(stream_name, text) for TTS
        self.clips = clips              # ClipStore, or None when not recording
        # callable(stream_name) -> bool; per-feed opt-in for clip recording.
        self.should_record = should_record or (lambda _name: False)
        # callable(stream_name) -> str; the Whisper prompt for THIS feed, so an
        # ATC feed isn't decoded with police vocabulary. Falls back to the global.
        self.prompt_for = prompt_for or (
            lambda _name: cfg.get("initial_prompt") or None)

    def run(self):
        while not self.stop_evt.is_set():
            try:
                name, color, audio, _ = self.jobq.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._transcribe(name, color, audio)
            except Exception as e:
                self.out.status(f"[{name}] transcription error: {e}")
            finally:
                self.jobq.task_done()

    def _transcribe(self, name, color, audio):
        segments, _info = self.model.transcribe(
            audio,
            language=self.cfg.get("language", "en"),
            beam_size=self.cfg.get("beam_size", 5),
            vad_filter=True,
            condition_on_previous_text=False,   # transmissions are independent
            initial_prompt=self.prompt_for(name) or None,
            no_speech_threshold=0.6,
            temperature=[0.0, 0.2, 0.4],
            # Anti-hallucination: drop repetition loops + low-confidence/garbage
            # segments, and stop the decoder repeating the same phrase.
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_repeat_ngram_size=3,
        )
        parts = []
        for s in segments:
            if getattr(s, "no_speech_prob", 0.0) > self.max_no_speech:
                continue
            if getattr(s, "avg_logprob", 0.0) < self.min_logprob:
                continue
            txt = s.text.strip()
            if txt and not _is_hallucination(txt, getattr(s, "no_speech_prob", 0.0)):
                parts.append(txt)
        text = _collapse_repeats(" ".join(parts).strip())
        if text:
            # Save the clip only once we know the segment produced real text:
            # hallucination-filtered transmissions would otherwise leave audio
            # on disk that no line ever points at.
            when = dt.datetime.now()
            clip_id = None
            if self.clips is not None and self.should_record(name):
                clip_id = self.clips.save(name, audio, text=text, when=when)
            self.out.line(name, color, text, clip_id=clip_id, when=when)
            if self.tts_hook:
                try:
                    self.tts_hook(name, text)
                except Exception:
                    pass


# --------------------------------------------------------------------------
# Engine: ties everything together and exposes a small API for any front-end
# (CLI or GUI). Loads the model once, manages stream workers dynamically, and
# owns the shared transcription queue + optional audio player.
# --------------------------------------------------------------------------
class Engine:
    def __init__(self, cfg, on_line=None, on_status=None,
                 console=True, file_logging=True, enable_audio=False):
        self.cfg = cfg
        self.out = Output(on_line=on_line, on_status=on_status,
                          console=console, file_logging=file_logging)
        self.ffmpeg = find_ffmpeg()
        self.vad_cfg = cfg.get("vad", {})
        self.jobq = queue.Queue(maxsize=200)
        self.stop_evt = threading.Event()
        self.model = None
        self.transcriber = None
        self.workers = {}               # name -> StreamWorker
        self._lock = threading.Lock()
        self.player = AudioPlayer() if enable_audio else None
        self.auth_header = self._build_auth()

        # Clip recording: which feeds save their audio (per-feed opt-in), plus
        # the store that writes/reads them. Populated by start_streams/add_stream
        # from each stream dict's "record" flag.
        self.clips = ClipStore(cfg, out=self.out, ffmpeg=self.ffmpeg)
        self.recording_feeds = set()
        # The stream dict per running feed, so per-feed settings (service
        # profile, prompt override) are resolvable by name at transcribe time.
        self.stream_meta = {}

        # Text-to-speech state (lazy: player created only when first enabled).
        tts = cfg.get("tts", {})
        self.tts = None
        self.tts_enabled = bool(tts.get("enabled", False))
        self.tts_engine = tts.get("engine", "auto")     # 'auto'|'piper'|'sapi'
        self.tts_voice = tts.get("voice")               # None -> first available
        self.tts_feeds = set(tts.get("feeds", []))      # stream names to speak
        self.tts_keywords = [k.lower() for k in tts.get("keywords", [])]
        self.tts_mode = tts.get("mode", "feeds")        # "feeds" | "keywords" | "both"
        self.tts_on_start = None    # callback(text) fired when an utterance begins
        self.tts_on_end = None      # callback(text) fired when it finishes

        # Purge old log files on startup (logs hold sensitive PII; keep bounded).
        if file_logging:
            days = cfg.get("log_retention_days")
            deleted = purge_old_logs(days)
            if deleted:
                self.out.status(f"Purged {len(deleted)} log file(s) older than {days} day(s).")
        # Same for clips -- voice recordings, so bounding them matters more.
        aged, over = apply_clip_retention(
            {"retention_days": self.clips.retention_days,
             "max_gb": self.clips.max_gb}, self.clips.dir)
        if aged:
            self.out.status(f"Purged {len(aged)} clip file(s) older than "
                            f"{self.clips.retention_days:g} day(s).")
        if over:
            self.out.status(f"Purged {len(over)} more clip file(s) to stay "
                            f"under {self.clips.max_gb:g} GB.")

    # -- auth ---------------------------------------------------------------
    def _build_auth(self):
        user, pw = load_credentials(self.cfg)
        if user and pw:
            self.out.status(f"Broadcastify auth: enabled (user '{user}')")
            return base64.b64encode(f"{user}:{pw}".encode()).decode()
        return None

    def apply_credentials(self, username, password, active_streams=None):
        """Persist new Broadcastify credentials, rebuild the auth header, and
        restart any running Broadcastify feeds so they reconnect with the new
        login (no app restart needed). `active_streams` is the caller's list of
        stream dicts (used to re-add restarted feeds); falls back to restarting
        by name only. Returns True if credentials were saved."""
        ok = save_credentials(username, password)
        self.auth_header = self._build_auth()
        # Restart running Broadcastify workers so they pick up the new header.
        # Only restart feeds we have a stream dict for (so we can re-add them);
        # leave pc-audio and non-Broadcastify streams untouched.
        by_name = {s["name"]: s for s in (active_streams or [])}
        running = set(self.stream_names())
        for name, stream in by_name.items():
            if name in running and is_broadcastify_stream(stream):
                self.remove_stream(name)
                self.add_stream(stream)
        return ok

    # -- lifecycle ----------------------------------------------------------
    def _make_whisper_model(self, model_name):
        """Build the transcription backend for `model_name`. On native ARM64 (or
        when cfg['engine'] selects it) this is the whisper.cpp backend; otherwise
        faster-whisper / ctranslate2. The device is resolved from cfg['device']
        ('auto'|'cuda'/'gpu'|'cpu'): GPU mode does first-run CUDA setup and, if the
        GPU can't actually be constructed, falls back to CPU rather than hard-
        failing (ctranslate2's only GPU backend is CUDA/NVIDIA -- a CPU-only or
        Intel Arc machine has no usable GPU here)."""
        if select_backend(self.cfg) == "whispercpp":
            return WhisperCppBackend(model_name, self.cfg, status_cb=self.out.status)

        # -- faster-whisper / ctranslate2 (x64; NVIDIA GPU or CPU) --
        # Allow tests / callers to inject a WhisperModel via module attribute;
        # otherwise import faster-whisper lazily (after CUDA is ready).
        WM = globals().get("WhisperModel")
        if WM is None:
            from faster_whisper import WhisperModel as WM

        device = str(self.cfg.get("device", "cuda")).strip().lower()
        want_gpu = device in ("", "auto", "cuda", "gpu")   # 'cpu' -> straight to CPU

        if want_gpu:
            ok, msg = ensure_cuda_libraries(status_cb=self.out.status)
            if not ok:
                self.out.status(msg + " Falling back to CPU (slower).")
                want_gpu = False
        add_nvidia_dll_dirs()

        if want_gpu:
            compute = self.cfg.get("compute_type", "float16")
            try:
                return WM(model_name, device="cuda", compute_type=compute)
            except Exception as e:
                # No usable CUDA device (CPU-only or Intel Arc machine, etc.).
                # Use CPU instead of crashing at load. Set "device": "cpu" in
                # config to skip this probe entirely.
                self.out.status(
                    f"GPU unavailable ({e}); using CPU (slower). "
                    "Tip: pick a smaller model (e.g. base.en) for good CPU speed."
                )
        return WM(model_name, device="cpu", compute_type="int8")

    def load_model(self):
        if select_backend(self.cfg) == "whispercpp":
            self.out.status(
                f"Loading Whisper '{self.cfg.get('model','large-v3')}' via "
                f"whisper.cpp (CPU/NPU) ..."
            )
        else:
            self.out.status(
                f"Loading Whisper '{self.cfg.get('model','large-v3')}' on "
                f"{self.cfg.get('device','cuda')}/{self.cfg.get('compute_type','float16')} ..."
            )
        t0 = time.time()
        self.model = self._make_whisper_model(self.cfg.get("model", "large-v3"))
        self.out.status(f"Model ready in {time.time()-t0:.1f}s.")
        self.clips.start()
        self.transcriber = Transcriber(self.model, self.cfg, self.jobq,
                                       self.out, self.stop_evt,
                                       clips=self.clips,
                                       should_record=self.is_recording,
                                       prompt_for=self.prompt_for)
        self.transcriber.tts_hook = self._maybe_speak
        self.transcriber.start()
        if self.tts_enabled:
            self._ensure_tts()

    # -- service profiles ---------------------------------------------------
    def profile_for(self, name):
        """The resolved service profile for a running feed (falls back to the
        default profile for feeds we have no dict for, e.g. after a restart)."""
        return service_profile(self.stream_meta.get(name, {}), self.cfg)

    def prompt_for(self, name):
        """The Whisper prompt for this feed: per-feed override, else its service
        preset, else the global initial_prompt."""
        return self.profile_for(name)["prompt"]

    # -- clip recording -----------------------------------------------------
    def is_recording(self, name):
        """True if this feed saves the audio behind each of its lines. Requires
        both the global switch and the feed's own opt-in."""
        return self.clips.enabled and name in self.recording_feeds

    def set_recording(self, name, on):
        """Turn clip recording on/off for one feed (takes effect immediately --
        the next transmission is saved or not, no restart)."""
        if on:
            self.recording_feeds.add(name)
        else:
            self.recording_feeds.discard(name)

    def set_clips_enabled(self, on):
        """Global switch. Off means no feed records, whatever its own flag."""
        self.clips.enabled = bool(on)
        if on:
            self.clips.start()

    def play_clip(self, clip_id, source=None):
        """Play a saved clip through the speakers. Returns True if audio started.
        Decoding runs on a worker thread so a click never blocks the UI.
        `source` overrides where the audio comes from -- anything with a
        load_pcm(), which is how an opened transcript bundle plays its own
        clips instead of the live store's."""
        if not (self.player and self.player.available and clip_id):
            return False
        store = source if source is not None else self.clips
        def _go():
            pcm = store.load_pcm(clip_id)
            if pcm:
                self.player.play_clip(pcm)
            else:
                self.out.status("That clip is no longer on disk.")
        threading.Thread(target=_go, daemon=True, name="clipplay").start()
        return True

    def stop_clip(self):
        if self.player:
            self.player.stop_clip()

    def clip_playing(self):
        """True while a saved clip is still sounding. Play-through polls this to
        know when to advance to the next line."""
        return bool(self.player and self.player.clip_playing())

    # -- read-along (playing a saved transcript through) ---------------------
    def speak_now(self, text):
        """Speak one line immediately, bypassing the feed/keyword rules that
        govern live TTS. Returns False if no voice is available."""
        if not self._ensure_tts():
            return False
        self.tts.say(text)
        return True

    def tts_busy(self):
        return bool(self.tts and self.tts.busy())

    def cancel_speech(self):
        """Stop the current utterance and drop the backlog (pause/stop)."""
        if self.tts:
            self.tts.cancel()

    # -- text-to-speech -----------------------------------------------------
    def _ensure_tts(self):
        """Create/start the TTS player if not running, or recreate it if the
        chosen voice or engine changed. Returns True if a working player is
        available."""
        engine = select_tts_engine({"engine": self.tts_engine})
        need_new = (self.tts is None or
                    (self.tts_voice and self.tts._voice_id != self.tts_voice) or
                    self.tts._engine != engine)
        if need_new:
            if self.tts is not None:
                self.tts.close()
            self.tts = TTSPlayer(voice_id=self.tts_voice, out=self.out,
                                 on_start=self.tts_on_start, on_end=self.tts_on_end,
                                 engine=engine)
            if self.tts.available:
                self.tts.start()
                self.out.status(f"TTS ready ({engine}).")
            else:
                self.out.status("TTS could not start (no voice / engine unavailable).")
        return bool(self.tts and self.tts.available)

    def set_tts_voice(self, voice_id):
        """Change the TTS voice (recreates the player on next _ensure_tts)."""
        self.tts_voice = voice_id

    def set_tts_engine(self, engine):
        """Change the TTS engine ('auto'|'piper'|'winrt'|'sapi'). If TTS is on,
        recreate the player so it takes effect (_ensure_tts recreates when the
        resolved engine changes)."""
        self.tts_engine = engine or "auto"
        if self.tts_enabled:
            self._ensure_tts()

    def _maybe_speak(self, name, text):
        """Decide whether this transcript line should be read aloud, and queue it."""
        if not self.tts_enabled or not self.tts or not self.tts.available:
            return
        speak = False
        if self.tts_mode in ("feeds", "both") and name in self.tts_feeds:
            speak = True
        if not speak and self.tts_mode in ("keywords", "both") and self.tts_keywords:
            if keyword_matches(text, self.tts_keywords):
                speak = True
        if speak:
            self.tts.say(text)

    def set_tts_enabled(self, enabled):
        self.tts_enabled = bool(enabled)
        if enabled:
            self._ensure_tts()
        elif self.tts:
            self.tts.set_muted(True)
        if self.tts:
            self.tts.set_muted(not enabled)

    def set_tts_feeds(self, names):
        self.tts_feeds = set(names or [])

    def set_tts_keywords(self, keywords):
        self.tts_keywords = [k.lower().strip() for k in (keywords or []) if k.strip()]

    def set_tts_mode(self, mode):
        if mode in ("feeds", "keywords", "both"):
            self.tts_mode = mode

    def tts_speak_test(self, text="Text to speech is working."):
        if self._ensure_tts():
            self.tts.say(text)

    def set_model(self, model_name, on_done=None, force=False):
        """
        Swap the Whisper model at runtime WITHOUT stopping the streams. Loads the
        new model (blocking -- call this from a background thread), then atomically
        hot-swaps the reference the transcriber reads. on_done(ok, message) is
        invoked when finished. Safe because the worker reads self.model once per
        transmission, so the reference swap takes effect on its next job.

        `force=True` reloads even when the model name is unchanged (used by
        set_device, which reloads the current model on a new device).
        """
        if model_name == self.cfg.get("model") and not force:
            if on_done:
                on_done(True, f"Already using '{model_name}'.")
            return
        self.out.status(f"Loading Whisper '{model_name}' (streams keep running)...")
        t0 = time.time()
        try:
            new_model = self._make_whisper_model(model_name)
        except Exception as e:
            msg = f"Model '{model_name}' failed to load: {e}"
            self.out.status(msg)
            if on_done:
                on_done(False, msg)
            return
        old = self.model
        self.model = new_model
        self.cfg["model"] = model_name
        if self.transcriber:
            self.transcriber.model = new_model   # atomic ref swap (GIL)
        del old
        msg = f"Switched to '{model_name}' in {time.time()-t0:.1f}s."
        self.out.status(msg)
        if on_done:
            on_done(True, msg)

    def set_device(self, device, on_done=None):
        """Change the compute device ('auto'|'cuda'/'gpu'|'cpu') and reload the
        current model live so it takes effect. No-op on the whisper.cpp backend
        (always CPU/NPU -- it ignores the device). Call from a background thread."""
        self.cfg["device"] = device
        if select_backend(self.cfg) == "whispercpp":
            if on_done:
                on_done(True, "Device is fixed to CPU/NPU on the whisper.cpp backend.")
            return
        self.set_model(self.cfg.get("model", "large-v3"), on_done=on_done, force=True)

    def start_streams(self, streams):
        # URL feeds are decoded by ffmpeg. If it's missing, say so ONCE here rather
        # than letting every worker loop on "ffmpeg launch failed / reconnecting".
        if any(s.get("type") not in ("pcaudio", "app") for s in streams
               if is_enabled(s)) and not ffmpeg_available(self.ffmpeg):
            self.out.status(
                "WARNING: ffmpeg not found -- URL/stream feeds cannot be decoded. "
                "Put ffmpeg.exe in the app's bin/ folder or install it on PATH.")
        for s in streams:
            if is_enabled(s):
                self.add_stream(s)

    # -- dynamic stream management -----------------------------------------
    def add_stream(self, stream):
        """Start a worker for a stream. Type 'pcaudio' captures a PC input
        device; anything else is a URL/feed via ffmpeg. Returns True if started."""
        name = stream["name"]
        # Clip recording is per-feed opt-in, carried on the stream dict.
        self.set_recording(name, stream.get("record", False))
        self.stream_meta[name] = dict(stream)     # for prompt_for / profile
        with self._lock:
            if name in self.workers:
                self.out.status(f"[{name}] already running.")
                return False
            if stream.get("type") == "app":
                w = ProcessLoopbackWorker(stream, self.vad_cfg, self.jobq,
                                          self.out, self.stop_evt)
            elif stream.get("type") == "pcaudio":
                # Prefer soundcard output-loopback (by device name); fall back to
                # the older Stereo Mix input-index capture if only `device` is set.
                if stream.get("output_device") is not None:
                    w = LoopbackWorker(stream, self.vad_cfg, self.jobq,
                                       self.out, self.stop_evt)
                else:
                    w = DeviceWorker(stream, self.vad_cfg, self.jobq,
                                     self.out, self.stop_evt)
            else:
                w = StreamWorker(stream, self.ffmpeg, self.vad_cfg, self.jobq,
                                 self.out, self.stop_evt, self.auth_header, self.player)
            self.workers[name] = w
            w.start()
        self.out.status(f"[{name}] added.")
        return True

    def remove_stream(self, name):
        with self._lock:
            w = self.workers.pop(name, None)
        self.recording_feeds.discard(name)
        self.stream_meta.pop(name, None)
        if w:
            w.stop()
            if self.player and self.player.get_source() == name:
                self.player.set_source(None)
            self.out.status(f"[{name}] removed.")
            return True
        return False

    def stream_names(self):
        with self._lock:
            return list(self.workers.keys())

    def change_device(self, stream, new_device):
        """Live-switch a pcaudio stream's capture device: stop its worker and
        start a fresh one on the new device. `new_device` is an output device
        NAME (soundcard) or, for legacy Stereo Mix streams, an input index.
        `stream` is the (mutated) config dict. Returns True if restarted."""
        name = stream["name"]
        if isinstance(new_device, str):
            stream["output_device"] = new_device      # soundcard loopback path
            stream.pop("device", None)
        else:
            stream["device"] = new_device             # legacy Stereo Mix index
        with self._lock:
            w = self.workers.pop(name, None)
        if w:
            w.stop()
        if is_enabled(stream):
            return self.add_stream(stream)
        return False

    # -- audio --------------------------------------------------------------
    def listen_to(self, name):
        """Make `name` audible (None mutes). No-op if audio is unavailable."""
        if self.player and self.player.available:
            self.player.set_source(name)
            return True
        return False

    def now_listening(self):
        return self.player.get_source() if self.player else None

    def audio_available(self):
        return bool(self.player and self.player.available)

    def shutdown(self):
        self.stop_evt.set()
        # Stop the clip writer FIRST: it drains what's queued, so clips from the
        # last few seconds still land instead of dying with the process.
        self.clips.stop()
        if self.player:
            self.player.close()
        if self.tts:
            self.tts.close()
        time.sleep(0.5)


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------
def main():
    enable_windows_ansi()
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH
    cfg = load_config(cfg_path)

    streams = [s for s in cfg.get("streams", []) if is_enabled(s)]
    if not streams:
        print("No enabled streams in config.json (add a 'url'). Nothing to do.")
        return

    engine = Engine(cfg, console=True, file_logging=True, enable_audio=False)
    if any("broadcastify.com" in s.get("url", "") or s.get("provider") == "broadcastify"
           for s in streams) and engine.auth_header is None:
        engine.out.status(
            "WARNING: a Broadcastify feed is configured but no credentials found. "
            "Add credentials.json or set BROADCASTIFY_USERNAME/PASSWORD."
        )
    engine.load_model()
    engine.start_streams(streams)
    engine.out.status(f"Listening to {len(streams)} stream(s). Press Ctrl+C to stop.\n")

    max_runtime = cfg.get("max_runtime_sec")  # optional; None = run forever
    started = time.time()
    try:
        while True:
            time.sleep(0.5)
            if max_runtime and (time.time() - started) >= max_runtime:
                engine.out.status(f"\nReached max_runtime_sec ({max_runtime}s); stopping.")
                break
    except KeyboardInterrupt:
        engine.out.status("\nStopping...")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
