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
import time
import queue
import base64
import threading
import subprocess
import collections
import datetime as dt


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
    # Prefer the ffmpeg bundled by the imageio-ffmpeg pip package (no PATH needed).
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fall back to a system ffmpeg on PATH


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
    In dev, everything stays in the project folder for convenience."""
    if getattr(sys, "frozen", False):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        d = os.path.join(base, "Transcriber")
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
    _seed_from_example(CONFIG_PATH, "config.example.json")
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
# Update check: compare installed faster-whisper / ctranslate2 against the
# latest on PyPI. Pure stdlib (urllib), short timeout, fails silently offline.
# Reports only -- it never installs anything (updating is a deliberate, manual
# `pip install -U ...` step, to avoid re-triggering the Python-version wheel trap).
# --------------------------------------------------------------------------
UPDATE_PACKAGES = ["faster-whisper", "ctranslate2"]


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


def check_for_updates(packages=UPDATE_PACKAGES, timeout=4.0):
    """
    Return a list of dicts, one per package:
      {"package", "installed", "latest", "update_available"}
    'latest' is None if PyPI couldn't be reached (offline / blocked).
    """
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


def interpreter_is_arm64():
    """True if this build is native ARM64 (so it should fetch the ARM64
    installer). Keys off the interpreter's OWN architecture, not the host's:
    an emulated x64 process on an ARM host must report False so it upgrades with
    the x64 installer. NOTE: platform.machine() can misreport under emulation on
    Windows, so we use the build/wheel platform tag and PROCESSOR_ARCHITECTURE
    (which reflect the process arch)."""
    import sysconfig
    plat = (sysconfig.get_platform() or "").lower()   # e.g. win-arm64 / win-amd64
    if "arm64" in plat or "aarch64" in plat:
        return True
    if "amd64" in plat or "x86_64" in plat or "win32" in plat:
        return False
    arch = (os.environ.get("PROCESSOR_ARCHITECTURE") or "").upper()
    return "ARM64" in arch


def _pick_installer_asset(assets):
    """Select the release .exe matching THIS build's architecture. Returns None
    if no matching-arch installer is present, so we report 'no update' rather
    than cross-installing the wrong architecture. Shared logic with the ARM
    build: each release carries both Transcriber-Setup-<v>.exe (x64) and
    Transcriber-ARM64-Setup-<v>.exe (arm64)."""
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
# Text-to-speech: reads selected transcript lines aloud in a clear neural voice
# (Piper). A background thread pulls text off a queue and synthesizes+plays it
# one utterance at a time so nothing overlaps. Lazy-loads Piper so the app runs
# fine without TTS. Speaks via sounddevice (separate stream from AudioPlayer).
# --------------------------------------------------------------------------
def list_tts_voices():
    """Return [(voice_id, path)] of downloaded Piper voices (*.onnx) in the voice
    dir. voice_id is the filename stem (e.g. 'en_US-lessac-medium')."""
    out = []
    if os.path.isdir(TTS_VOICE_DIR):
        for p in sorted(glob.glob(os.path.join(TTS_VOICE_DIR, "*.onnx"))):
            out.append((os.path.splitext(os.path.basename(p))[0], p))
    return out


def tts_available():
    try:
        import piper  # noqa: F401
        return len(list_tts_voices()) > 0
    except Exception:
        return False


class TTSPlayer(threading.Thread):
    """Background speech queue: put(text) -> spoken aloud, one at a time.
    Drops items if the backlog grows (so it never lags far behind live audio).
    Optional on_start(text)/on_end(text) callbacks fire around each utterance
    (used by the GUI to highlight the line currently being read)."""
    def __init__(self, voice_id=None, max_queue=6, out=None,
                 on_start=None, on_end=None):
        self.on_start = on_start
        self.on_end = on_end
        super().__init__(daemon=True, name="tts")
        self.q = queue.Queue(maxsize=max_queue)
        self.stop_evt = threading.Event()
        self.out = out
        self._voice = None
        self._voice_id = voice_id
        self._sr = 22050
        self._ok = self._load_voice(voice_id)
        self._muted = False

    def _load_voice(self, voice_id):
        try:
            from piper import PiperVoice
        except Exception as e:
            if self.out:
                self.out.status(f"TTS unavailable (piper not installed): {e}")
            return False
        voices = list_tts_voices()
        if not voices:
            if self.out:
                self.out.status("TTS: no voice models in tts_voices/.")
            return False
        path = dict(voices).get(voice_id) or voices[0][1]
        self._voice_id = voice_id or voices[0][0]
        try:
            self._voice = PiperVoice.load(path)
            self._sr = self._voice.config.sample_rate
            return True
        except Exception as e:
            if self.out:
                self.out.status(f"TTS: failed to load voice: {e}")
            return False

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
        import numpy as _np
        try:
            import sounddevice as sd
        except Exception:
            return
        while not self.stop_evt.is_set():
            try:
                text = self.q.get(timeout=0.3)
            except queue.Empty:
                continue
            if self._muted or not self._voice:
                continue
            try:
                chunks = [
                    _np.frombuffer(c.audio_int16_bytes, dtype=_np.int16)
                    for c in self._voice.synthesize(text)
                ]
                if not chunks:
                    continue
                audio = _np.concatenate(chunks)
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
    try:
        import proctap  # noqa: F401
        return True
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
# Transcription worker: single shared GPU model serving all streams
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
        """Create a WhisperModel, handling first-run CUDA setup for GPU mode:
        (1) download the CUDA runtime if the slim installer left it out,
        (2) register the DLL directories, (3) lazily import + construct."""
        device = self.cfg.get("device", "cuda")
        if device == "cuda":
            ok, msg = ensure_cuda_libraries(status_cb=self.out.status)
            if not ok:
                self.out.status(msg + " Falling back to CPU (slower).")
                device = "cpu"
        add_nvidia_dll_dirs()
        # Allow tests / callers to inject a WhisperModel via module attribute;
        # otherwise import faster-whisper lazily (after CUDA is ready).
        WM = globals().get("WhisperModel")
        if WM is None:
            from faster_whisper import WhisperModel as WM
        compute = self.cfg.get("compute_type", "float16") if device == "cuda" else "int8"
        return WM(model_name, device=device, compute_type=compute)

    def load_model(self):
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
        chosen voice changed. Returns True if a working player is available."""
        need_new = (self.tts is None or
                    (self.tts_voice and self.tts._voice_id != self.tts_voice))
        if need_new:
            if self.tts is not None:
                self.tts.close()
            self.tts = TTSPlayer(voice_id=self.tts_voice, out=self.out,
                                 on_start=self.tts_on_start, on_end=self.tts_on_end)
            if self.tts.available:
                self.tts.start()
                self.out.status(f"TTS ready (voice '{self.tts._voice_id}').")
            else:
                self.out.status("TTS could not start (no voice / piper).")
        return bool(self.tts and self.tts.available)

    def set_tts_voice(self, voice_id):
        """Change the TTS voice (recreates the player on next _ensure_tts)."""
        self.tts_voice = voice_id

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

    def set_model(self, model_name, on_done=None):
        """
        Swap the Whisper model at runtime WITHOUT stopping the streams. Loads the
        new model (blocking -- call this from a background thread), then atomically
        hot-swaps the reference the transcriber reads. on_done(ok, message) is
        invoked when finished. Safe because the worker reads self.model once per
        transmission, so the reference swap takes effect on its next job.
        """
        if model_name == self.cfg.get("model"):
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

    def start_streams(self, streams):
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
