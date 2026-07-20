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


def extract_callsign(text, extra_prefixes=None):
    """
    Return a normalized unit call sign found in `text` (e.g. "ADAM 33",
    "ENGINE 14") or None. High precision: rejects spelled-out plates and street
    addresses. `extra_prefixes` (iterable of lowercase words) extends the set of
    recognized unit prefixes for local department lingo.

    NOTE: this identifies the FIRST unit MENTIONED in a transmission, which is a
    heuristic for who is involved -- not a guaranteed acoustic speaker ID.
    """
    if not text:
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
_ADDR_NUMBERED = re.compile(
    r"\b(\d{2,6})\s+"
    r"((?:" + _DIRS + r"\s+)?" + _NAME_NT + r"(?:\s+" + _NAME_NT + r"){0,2})"
    r"(?:\s+(" + _STREET_TYPE_RE + r"))?\b",
    re.I)

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
        self.on_line = on_line          # callback(stream_name, color, text, ts)
        self.on_status = on_status      # callback(msg)
        self.console = console
        self.file_logging = file_logging
        if self.file_logging:
            os.makedirs(LOG_DIR, exist_ok=True)

    def line(self, stream_name, color, text):
        ts = dt.datetime.now().strftime("%H:%M:%S")
        with self._lock:
            if self.console:
                code = ANSI_COLORS.get(color, "97")
                print(f"\033[{code}m[{ts}] {stream_name:<10}\033[0m {text}", flush=True)
            if self.file_logging:
                day = dt.datetime.now().strftime("%Y%m%d")
                path = os.path.join(LOG_DIR, f"{safe_filename(stream_name)}-{day}.log")
                with open(path, "a", encoding="utf-8") as f:
                    f.write(f"[{ts}] {text}\n")
        if self.on_line:
            try:
                self.on_line(stream_name, color, text, ts)
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
# --------------------------------------------------------------------------
class AudioPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._source = None             # name of the stream currently audible
        self._stream = None
        self._sd = None
        self._buf = bytearray()
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
            have = min(need, len(self._buf))
            outdata[:have] = bytes(self._buf[:have])
            del self._buf[:have]
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

    def feed(self, name, pcm_bytes):
        if not self._ok:
            return
        with self._lock:
            if name != self._source:
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

    @property
    def available(self):
        return self._ok

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
                sd.play(audio, self._sr)
                # Wait for playback, but bail out promptly on stop/mute.
                while sd.get_stream().active and not self.stop_evt.is_set() \
                        and not self._muted:
                    time.sleep(0.05)
                if self._muted or self.stop_evt.is_set():
                    sd.stop()
            except Exception as e:
                if self.out:
                    self.out.status(f"TTS error: {e}")
            finally:
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


def list_output_devices():
    """Return [(name, is_default)] of output devices that can be loopback-captured
    via soundcard. Empty if soundcard is unavailable. Names are de-duplicated."""
    try:
        import soundcard as sc
    except Exception:
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
    build, so `import proctap` succeeds there while every capture raises. Require
    the native extension too, otherwise the GUI would offer an 'application' source
    that always fails."""
    try:
        import proctap  # noqa: F401
        import importlib.util
        return importlib.util.find_spec("proctap._native") is not None
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
        if status_cb:
            status_cb(f"Loading whisper.cpp model '{wname}' "
                      f"({self._n_threads} threads)...")
        self._model = Model(
            model=wname, models_dir=models_dir,
            redirect_whispercpp_logs_to=False,
            n_threads=self._n_threads, print_progress=False,
            print_realtime=False,
        )

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
    def __init__(self, model, cfg, jobq, out, stop_evt):
        super().__init__(daemon=True, name="transcriber")
        self.model = model
        self.cfg = cfg
        self.jobq = jobq
        self.out = out
        self.stop_evt = stop_evt
        self.max_no_speech = cfg.get("filters", {}).get("max_no_speech_prob", 0.6)
        self.min_logprob = cfg.get("filters", {}).get("min_avg_logprob", -1.0)
        self.tts_hook = None    # optional callable(stream_name, text) for TTS

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
            initial_prompt=self.cfg.get("initial_prompt") or None,
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
            self.out.line(name, color, text)
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
        self.transcriber = Transcriber(self.model, self.cfg, self.jobq,
                                       self.out, self.stop_evt)
        self.transcriber.tts_hook = self._maybe_speak
        self.transcriber.start()
        if self.tts_enabled:
            self._ensure_tts()

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
