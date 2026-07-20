"""
Tkinter GUI for the stream transcriber.

Features:
  - Menu / toolbar to add and remove streams (with provider + color).
  - Two view modes:
        * Unified  - one combined feed, each line tagged with its sector.
        * Sectors  - a separate scrolling panel per stream, side by side.
  - "Listen" lets you hear ONE stream at a time through your speakers while all
    streams keep transcribing (you stay muted on the others).
  - Streams + settings are saved to config.json.

Launch via gui.bat, or:
    .venv\\Scripts\\python.exe -E gui.py

Threading note: faster-whisper / stream workers call back from background
threads. Tkinter is single-threaded, so those callbacks only enqueue events;
the UI thread drains the queue via .after() and does all widget updates.
"""
import os
import queue
import threading
import collections
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import tkinter.font as tkfont

import transcriber as core

# Branding assets (optional; splash/icons are skipped gracefully if missing).
# Use the same resource dir as the engine so this works both in dev and when
# frozen into an installed .exe (assets live next to the executable).
HERE = core.HERE
SPLASH_LOGO = os.path.join(HERE, "OfficialLogo.png")
TASKBAR_ICON = os.path.join(HERE, "OfficialTaskbarIcon.png")
DEVELOPER_PHOTO = os.path.join(HERE, "Developer.png")

APP_VERSION = "1.4"


def load_scaled_image(path, max_w=None, max_h=None):
    """Load a PNG as a Tk PhotoImage, optionally scaled to fit within max_w/max_h
    while preserving aspect ratio. Uses Pillow's high-quality LANCZOS resampling
    when available (crisp); falls back to Tk subsample (blocky) if not. Returns a
    PhotoImage (keep a reference!) or None on failure."""
    try:
        from PIL import Image, ImageTk
        im = Image.open(path)
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGBA")
        w, h = im.size
        if max_w or max_h:
            scale = min((max_w / w) if max_w else 1.0, (max_h / h) if max_h else 1.0)
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                               Image.LANCZOS)
        return ImageTk.PhotoImage(im)
    except Exception:
        # Fallback: Tk-only, integer subsample (lower quality).
        try:
            img = tk.PhotoImage(file=path)
            w, h = img.width(), img.height()
            factor = 1
            if max_w:
                while (w // (factor + 1)) >= max_w and factor < 20:
                    factor += 1
            if max_h:
                while (h // (factor + 1)) >= max_h and factor < 20:
                    factor += 1
            return img.subsample(factor, factor) if factor > 1 else img
        except Exception:
            return None


# Transcript font sizing.
FONT_FAMILY = "Consolas"
FONT_MIN, FONT_MAX, FONT_DEFAULT = 8, 28, 11

# How many recent transcript lines to keep in memory for replay after the view
# is rebuilt (toggling a feed, switching views). Disk logs are the full record.
HISTORY_MAX = 2000

# A spread of distinct, readable hues for coloring by unit/call sign. Units are
# mapped to these deterministically so the same unit keeps the same color all
# session (and across view rebuilds).
UNIT_PALETTE = [
    "#e06c75", "#98c379", "#e5c07b", "#61afef", "#c678dd", "#56b6c2",
    "#d19a66", "#56a8f5", "#e88ac0", "#7ec699", "#c0a36e", "#a0a8f0",
    "#f08d6f", "#6fcfd0", "#b58ee0", "#9fc16b",
]

# Tk color hex for each named color the engine uses.
COLOR_HEX = {
    "red": "#e06c75", "green": "#98c379", "yellow": "#e5c07b",
    "blue": "#61afef", "magenta": "#c678dd", "cyan": "#56b6c2",
    "white": "#e6e6e6", "grey": "#9aa0a6",
}
COLOR_CHOICES = list(COLOR_HEX.keys())

BG = "#1e1f22"
BG2 = "#26282c"
FG = "#e6e6e6"
MUTED = "#9aa0a6"
NO_UNIT_COLOR = FG          # white for lines with no detected speaker/call sign
LINK_FG = "#6db3f2"         # blue for clickable address -> Google Maps links

# Read-aloud keyword presets: each checkbox expands to several synonyms so you
# catch variants without typing them all. Label -> list of match terms (lower).
KEYWORD_PRESETS = [
    ("Shooting", ["shots", "gunshots", "shooting", "gunfire", "gun", "shots fired"]),
    ("Fire", ["fire", "structure fire", "smoke", "flames"]),
    ("Suspect", ["suspect", "suspect description"]),
    ("Crash", ["crash", "accident", "collision", "mvc", "rollover"]),
    ("Hostage", ["hostage", "barricade", "barricaded"]),
    ("Pursuit", ["pursuit", "chase", "fleeing", "foot pursuit"]),
    ("Officer needs help", ["officer down", "officer needs assistance", "backup",
                            "shots at officer", "10-33", "signal zero"]),
    ("Weapon", ["weapon", "knife", "armed", "stabbing", "firearm"]),
    ("Medical", ["cardiac", "not breathing", "cpr", "overdose", "unresponsive", "od"]),
    ("Robbery/Burglary", ["robbery", "burglary", "break-in", "breaking and entering"]),
    ("Assault", ["assault", "fight", "battery", "domestic"]),
    ("Missing person", ["missing", "amber alert", "missing person", "silver alert"]),
    ("Explosion", ["explosion", "blast", "detonation"]),
    ("Injury", ["injured", "trauma", "victim down", "bleeding"]),
]


def expand_keyword_presets(preset_labels):
    """Return the flat list of match terms for the given checked preset labels."""
    m = dict(KEYWORD_PRESETS)
    terms = []
    for lab in preset_labels:
        terms.extend(m.get(lab, []))
    return terms


class CheckBox(tk.Label):
    """A themeable checkbox: a clickable ☑/☐ glyph whose color we fully control.
    tk.Checkbutton's tick is OS-drawn on Windows and won't recolor reliably, so
    we roll our own. .get()/.set() mirror a BooleanVar-like API; pass `command`
    for a toggle callback. `disabled=True` shows a dimmed, non-clickable box."""
    BOX_ON = "☑"     # ☑
    BOX_OFF = "☐"    # ☐

    def __init__(self, parent, value=False, command=None, disabled=False, **kw):
        self._value = bool(value)
        self._command = command
        self._disabled = disabled
        super().__init__(parent, bg=kw.pop("bg", BG),
                         fg=(MUTED if disabled else FG),
                         font=kw.pop("font", ("Segoe UI", 12)), cursor="arrow", **kw)
        self._refresh()
        if not disabled:
            self.configure(cursor="hand2")
            self.bind("<Button-1>", self._on_click)

    def _refresh(self):
        self.config(text=self.BOX_ON if self._value else self.BOX_OFF,
                    fg=(MUTED if self._disabled else FG))

    def _on_click(self, _e=None):
        if self._disabled:
            return
        self._value = not self._value
        self._refresh()
        if self._command:
            self._command(self._value)

    def get(self):
        return self._value

    def set(self, v):
        self._value = bool(v)
        self._refresh()

# Whisper models selectable in the GUI. Larger = more accurate; distil/turbo are
# faster. Switching downloads the model once (cached) and hot-swaps live.
MODEL_CHOICES = ["large-v3", "large-v3-turbo", "distil-large-v3", "medium", "small"]

# Built-in catalog of feeds verified working this session (Greater Cleveland).
# Users pick from these in "Add from catalog..." instead of pasting URLs.
FEED_CATALOG = [
    {"name": "Cleveland West", "url": "https://www.broadcastify.com/listen/feed/25008",
     "color": "cyan", "provider": "broadcastify", "location": "Cleveland, OH",
     "desc": "Cleveland Police - West (1st & 2nd District)"},
    {"name": "Cleveland Citywide", "url": "https://www.broadcastify.com/listen/feed/11446",
     "color": "green", "provider": "broadcastify", "location": "Cleveland, OH",
     "desc": "Cleveland Police + Metro Housing (citywide, covers east side)"},
    {"name": "Cleveland Fire/EMS", "url": "https://www.broadcastify.com/listen/feed/23058",
     "color": "red", "provider": "broadcastify", "location": "Cleveland, OH",
     "desc": "Cleveland Fire and EMS"},
    {"name": "Westlake/WestCom", "url": "https://www.broadcastify.com/listen/feed/15234",
     "color": "yellow", "provider": "broadcastify", "location": "Westlake, OH",
     "desc": "WestCom: Westlake, Bay Village, Fairview Park, Rocky River, N. Olmsted (PD+Fire)"},
    {"name": "East Cleveland", "url": "https://www.broadcastify.com/listen/feed/42707",
     "color": "magenta", "provider": "broadcastify", "location": "East Cleveland, OH",
     "desc": "East Cleveland Police and Fire Dispatch"},
]


class AddStreamDialog(simpledialog.Dialog):
    """Modal dialog to add (or edit) a stream. Source can be:
      - url: a stream/feed URL
      - pc audio: capture an output device (all sound from those speakers)
      - application: capture ONE app's audio by process (per-app loopback).
    Pass `initial` (a stream dict) to prefill for editing."""
    def __init__(self, parent, title="Add stream", initial=None):
        self._initial = initial or {}
        super().__init__(parent, title=title)

    def body(self, master):
        self.configure(bg=BG)
        master.configure(bg=BG)
        init = self._initial

        tk.Label(master, text="Name (sector label)", bg=BG, fg=FG).grid(
            row=0, column=0, sticky="w", padx=6, pady=4)
        self.name_var = tk.StringVar(value=init.get("name", ""))
        tk.Entry(master, textvariable=self.name_var, width=44).grid(
            row=0, column=1, padx=6, pady=4)

        # Source type. "application" only offered if per-app capture is available.
        tk.Label(master, text="Source", bg=BG, fg=FG).grid(
            row=1, column=0, sticky="w", padx=6, pady=4)
        init_source = {"pcaudio": "pc audio", "app": "application"}.get(
            init.get("type"), "url")
        self.source = tk.StringVar(value=init_source)
        sources = ["url", "pc audio"]
        if core.proctap_available():
            sources.append("application")
        ttk.Combobox(master, textvariable=self.source, values=sources,
                     state="readonly", width=20).grid(row=1, column=1, sticky="w", padx=6)
        self.source.trace_add("write", lambda *_: self._sync_state())

        # URL row.
        self.url_label = tk.Label(master, text="Stream URL or feed id/page",
                                  bg=BG, fg=FG)
        self.url_label.grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.url_var = tk.StringVar(value=init.get("url", ""))
        self.url_entry = tk.Entry(master, textvariable=self.url_var, width=44)
        self.url_entry.grid(row=2, column=1, padx=6, pady=4)

        # PC-audio device row: pick the OUTPUT device (speakers) to capture.
        self.dev_label = tk.Label(master, text="Speakers to capture", bg=BG, fg=FG)
        self.dev_label.grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self._outputs = core.list_output_devices()   # [(name, is_default)]
        out_names = [n for n, _d in self._outputs] or ["(none found)"]
        self.dev_var = tk.StringVar()
        self.dev_combo = ttk.Combobox(master, textvariable=self.dev_var,
                                      values=out_names, state="disabled", width=42)
        self.dev_combo.grid(row=3, column=1, padx=6, pady=4)
        if init.get("output_device") in out_names:        # prefill when editing
            self.dev_var.set(init["output_device"])
        else:
            for n, is_def in self._outputs:
                if is_def:
                    self.dev_var.set(n)
                    break
            if not self.dev_var.get() and out_names and out_names[0] != "(none found)":
                self.dev_var.set(out_names[0])

        # Application row: pick a running app that has audio. Refresh re-scans.
        self.app_label = tk.Label(master, text="Application (playing audio)",
                                  bg=BG, fg=FG)
        self.app_label.grid(row=4, column=0, sticky="w", padx=6, pady=4)
        approw = tk.Frame(master, bg=BG)
        approw.grid(row=4, column=1, sticky="w", padx=6)
        self.app_var = tk.StringVar()
        self.app_combo = ttk.Combobox(approw, textvariable=self.app_var,
                                      values=["(none)"], state="disabled", width=34)
        self.app_combo.pack(side="left")
        self.app_refresh = tk.Button(approw, text="↻", width=2, command=self._refresh_apps,
                                     bg=BG2, fg=FG, relief="flat", state="disabled")
        self.app_refresh.pack(side="left", padx=(4, 0))
        self._apps = []            # [(pid, name, active)]

        tk.Label(master, text="Color", bg=BG, fg=FG).grid(
            row=5, column=0, sticky="w", padx=6, pady=4)
        self.color = tk.StringVar(value=init.get("color", "cyan"))
        ttk.Combobox(master, textvariable=self.color, values=COLOR_CHOICES,
                     state="readonly", width=20).grid(row=5, column=1, sticky="w", padx=6)

        # Location: city/state to anchor clickable address map-links for this feed.
        tk.Label(master, text="Location (for map links)", bg=BG, fg=FG).grid(
            row=6, column=0, sticky="w", padx=6, pady=4)
        self.location = tk.StringVar(value=init.get("location", ""))
        tk.Entry(master, textvariable=self.location, width=30).grid(
            row=6, column=1, sticky="w", padx=6, pady=4)
        tk.Label(master, text="e.g. Cleveland, OH", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8)).grid(row=7, column=1, sticky="w", padx=6)

        self.provider = tk.StringVar(value=init.get("provider", "broadcastify"))
        self._sync_state()
        return None

    def _refresh_apps(self):
        self._apps = core.list_audio_apps()
        labels = [f"{name} (pid {pid}){'  ● playing' if active else ''}"
                  for pid, name, active in self._apps]
        if not labels:
            labels = ["(no apps playing audio — start playback, then ↻)"]
        self.app_combo.config(values=labels)
        if self._apps:
            self.app_var.set(labels[0])     # first = currently-playing
        else:
            self.app_var.set(labels[0])

    def _sync_state(self):
        """Enable only the fields relevant to the chosen source type."""
        src = self.source.get()
        is_url = src == "url"
        is_pc = src == "pc audio"
        is_app = src == "application"
        self.url_entry.config(state="normal" if is_url else "disabled")
        self.url_label.config(fg=FG if is_url else MUTED)
        self.dev_combo.config(state="readonly" if is_pc else "disabled")
        self.dev_label.config(fg=FG if is_pc else MUTED)
        self.app_combo.config(state="readonly" if is_app else "disabled")
        self.app_refresh.config(state="normal" if is_app else "disabled")
        self.app_label.config(fg=FG if is_app else MUTED)
        if is_app and not self._apps:
            self._refresh_apps()

    def validate(self):
        if not self.name_var.get().strip():
            messagebox.showwarning("Missing info", "Name is required.")
            return False
        src = self.source.get()
        if src == "pc audio":
            if not self._outputs or not self.dev_var.get():
                messagebox.showwarning("Missing info", "Pick the speakers to capture.")
                return False
        elif src == "application":
            if not self._apps:
                messagebox.showwarning("No app", "No application is playing audio. "
                                       "Start playback and click ↻.")
                return False
            if not self.app_var.get() or "pid" not in self.app_var.get():
                messagebox.showwarning("Missing info", "Pick an application.")
                return False
        elif not self.url_var.get().strip():
            messagebox.showwarning("Missing info", "A stream URL is required.")
            return False
        return True

    def _selected_app(self):
        """Return (pid, name) for the chosen app combo entry, or (None, None)."""
        sel = self.app_var.get()
        for pid, name, _active in self._apps:
            if f"(pid {pid})" in sel:
                return pid, name
        return None, None

    def apply(self):
        src = self.source.get()
        if src == "pc audio":
            self.result = {
                "name": self.name_var.get().strip(), "type": "pcaudio",
                "output_device": self.dev_var.get(), "color": self.color.get(),
            }
        elif src == "application":
            pid, app_name = self._selected_app()
            self.result = {
                "name": self.name_var.get().strip(), "type": "app",
                "pid": pid, "app_name": app_name, "color": self.color.get(),
            }
        else:
            self.result = {
                "name": self.name_var.get().strip(),
                "url": self.url_var.get().strip(),
                "provider": self.provider.get(), "color": self.color.get(),
            }
        loc = self.location.get().strip()
        if loc:
            self.result["location"] = loc


class ChangeDeviceDialog(simpledialog.Dialog):
    """Pick which speakers (output device) to capture for a PC-audio stream.
    'Detect signal' samples each output's live loopback level so the user can see
    which one their audio is actually playing through. result = output device name."""
    def __init__(self, parent, current_device):
        self._current = current_device                   # output device NAME
        self._outputs = core.list_output_devices()       # [(name, is_default)]
        self._levels = {}                                # name -> rms
        super().__init__(parent, title="Change audio source")

    def body(self, master):
        self.configure(bg=BG)
        master.configure(bg=BG)
        tk.Label(master, text="Pick the speakers to capture:", bg=BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(8, 2))
        tk.Label(master, text="Tip: with your audio (e.g. the TV) playing, click "
                 "'Detect signal' — the device with the\nhighest level is the one "
                 "your sound is coming from.", bg=BG, fg=MUTED, justify="left").pack(
                     anchor="w", padx=8, pady=(0, 6))

        self.listbox = tk.Listbox(master, width=58, height=8, bg="#16171a",
                                  fg=FG, selectbackground="#3a3d44",
                                  activestyle="none", exportselection=False)
        self.listbox.pack(fill="both", expand=True, padx=8)
        self._refresh_list()

        btns = tk.Frame(master, bg=BG)
        btns.pack(fill="x", padx=8, pady=(6, 2))
        self.detect_btn = tk.Button(btns, text="🔊 Detect signal",
                                    command=self._detect, bg=BG2, fg=FG, relief="flat")
        self.detect_btn.pack(side="left")
        self.detect_status = tk.Label(btns, text="", bg=BG, fg=MUTED)
        self.detect_status.pack(side="left", padx=8)
        return self.listbox

    def _label_for(self, name, is_def):
        mark = "● " if name == self._current else "   "
        tag = "  (default)" if is_def else ""
        lvl = self._levels.get(name)
        meter = ""
        if lvl is not None:
            bars = int(min(lvl, 0.05) / 0.05 * 12)
            meter = "  " + ("█" * bars + "·" * (12 - bars))
        return f"{mark}🔊 {name}{tag}{meter}"

    def _refresh_list(self, select_name=None):
        self.listbox.delete(0, "end")
        self._rows = []
        for name, is_def in self._outputs:
            self.listbox.insert("end", self._label_for(name, is_def))
            self._rows.append(name)
        want = select_name if select_name is not None else self._current
        if want in self._rows:
            pos = self._rows.index(want)
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(pos)
            self.listbox.see(pos)

    def _detect(self):
        self.detect_btn.config(state="disabled")
        self.detect_status.config(text="Listening to each output...")
        self.update_idletasks()
        levels = []
        try:
            levels = core.probe_output_levels(0.5)
            self._levels = {name: lvl for name, lvl, _d in levels}
        finally:
            self.detect_btn.config(state="normal")
        loud = [r for r in levels if r[1] > 0.0005]
        if loud:
            best = max(loud, key=lambda r: r[1])[0]
            self._refresh_list(select_name=best)
            self.detect_status.config(text="Selected the output with signal.")
        else:
            self._refresh_list()
            self.detect_status.config(text="No signal — is audio playing?")

    def validate(self):
        if not self.listbox.curselection():
            messagebox.showwarning("Pick one", "Select a device.", parent=self)
            return False
        return True

    def apply(self):
        self.result = self._rows[self.listbox.curselection()[0]]


class CatalogDialog(tk.Toplevel):
    """The single feed manager. Lists every saved feed (built-in + user-added).
    Per row: Add (start transcribing) / Remove (stop, stays saved) / Edit /
    Delete (forget). 'Add new feed' saves to the library without auto-starting.
    Operates on app.library + app.streams directly."""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("Feeds")
        self.geometry("600x460")
        self.configure(bg=BG)
        self.transient(parent)

        tk.Label(self, text="Feeds — Add starts transcribing, Remove stops it, "
                 "Edit / Delete to manage",
                 bg=BG, fg=FG, font=("Segoe UI", 10, "bold")).pack(
                     anchor="w", padx=10, pady=(10, 4))

        self.listframe = tk.Frame(self, bg=BG)
        self.listframe.pack(fill="both", expand=True, padx=10)

        btns = tk.Frame(self, bg=BG2)
        btns.pack(side="bottom", fill="x")
        tk.Button(btns, text="+ Add new feed", command=self._add_new,
                  bg=BG2, fg=FG, relief="flat").pack(side="left", padx=4, pady=6)
        tk.Button(btns, text="Close", command=self.destroy,
                  bg=BG2, fg=FG, relief="flat").pack(side="right", padx=4, pady=6)

        self._render()

    def _render(self):
        for w in self.listframe.winfo_children():
            w.destroy()
        # "Active" = currently transcribing (in streams AND enabled). A feed that
        # was removed or toggled Off is NOT active and can be (re-)added.
        active_names = {s["name"] for s in self.app.streams if core.is_enabled(s)}

        hdr = tk.Frame(self.listframe, bg=BG)
        hdr.pack(fill="x", pady=(0, 4))
        tk.Label(hdr, text="   Feed  (drag ⠿ to reorder)", bg=BG, fg=MUTED,
                 anchor="w").pack(side="left", fill="x", padx=(6, 0))

        if not self.app.library:
            tk.Label(self.listframe, text="(library empty — click 'Add new feed')",
                     bg=BG, fg=MUTED).pack(anchor="w", pady=8)
            return

        self._rows = []   # (row_frame, name) for drag hit-testing
        for entry in self.app.library:
            name = entry["name"]
            is_active = name in active_names
            row = tk.Frame(self.listframe, bg=BG)
            row.pack(fill="x", pady=1)
            self._rows.append((row, name))
            grip = tk.Label(row, text="⠿", bg=BG, fg=MUTED, cursor="fleur",
                            font=("Segoe UI", 11))
            grip.pack(side="left", padx=(6, 0))
            grip.bind("<ButtonPress-1>", lambda e, n=name: self._drag_start(n))
            grip.bind("<B1-Motion>", self._drag_motion)
            grip.bind("<ButtonRelease-1>", self._drag_drop)
            tk.Label(row, text=f"  {name}", bg=BG,
                     fg=COLOR_HEX.get(entry.get("color", "white"), FG),
                     anchor="w", font=("Segoe UI", 10), width=28).pack(side="left")
            tk.Button(row, text="Delete", command=lambda n=name: self._delete(n),
                      bg=BG2, fg=FG, relief="flat").pack(side="right", padx=2)
            tk.Button(row, text="Edit", command=lambda n=name: self._edit(n),
                      bg=BG2, fg=FG, relief="flat").pack(side="right", padx=2)
            # Toggle action: active feeds get "Remove" (stop transcribing); the
            # rest get "Add" (start). Both keep the feed in the library.
            if is_active:
                tk.Button(row, text="Remove", command=lambda n=name: self._remove_one(n),
                          bg=BG2, fg=FG, relief="flat").pack(side="right", padx=2)
                tk.Label(row, text="active", bg=BG, fg="#98c379",
                         width=6).pack(side="right", padx=2)
            else:
                tk.Button(row, text="Add", command=lambda n=name: self._add_one(n),
                          bg=BG2, fg=FG, relief="flat").pack(side="right", padx=2)

    # ----- drag-to-reorder rows -------------------------------------------
    def _drag_start(self, name):
        self._drag_name = name
        self._drag_shadow = None

    def _row_at(self, y_root):
        """Return the feed name of the row under the pointer's y, or None."""
        for row, nm in getattr(self, "_rows", []):
            try:
                top = row.winfo_rooty()
                bot = top + row.winfo_height()
            except Exception:
                continue
            if top <= y_root <= bot:
                return nm
        return None

    def _drag_motion(self, event):
        name = getattr(self, "_drag_name", None)
        if not name:
            return
        if not getattr(self, "_drag_shadow", None):
            sh = tk.Toplevel(self)
            sh.overrideredirect(True)
            sh.attributes("-topmost", True)
            try:
                sh.attributes("-alpha", 0.85)
            except Exception:
                pass
            tk.Label(sh, text=f"⠿ {name}", bg=BG2, fg=FG,
                     font=("Segoe UI", 10, "bold"), padx=10, pady=3,
                     relief="solid", borderwidth=1).pack()
            self._drag_shadow = sh
        self._drag_shadow.geometry(f"+{event.x_root + 12}+{event.y_root + 8}")
        # Highlight the row under the pointer.
        over = self._row_at(event.y_root)
        for row, nm in self._rows:
            row.config(bg="#3a3d44" if (nm == over and nm != name) else BG)

    def _drag_drop(self, event):
        name = getattr(self, "_drag_name", None)
        self._drag_name = None
        sh = getattr(self, "_drag_shadow", None)
        if sh is not None:
            try:
                sh.destroy()
            except Exception:
                pass
        self._drag_shadow = None
        target = self._row_at(event.y_root)
        if name and target and target != name:
            self.app._reorder_library(name, target)   # persists to config
        self._render()

    def _add_new(self):
        # New feeds go to the LIBRARY ONLY -- they don't start transcribing until
        # the user clicks "Add" on the row.
        dlg = AddStreamDialog(self, title="Add new feed")
        if dlg.result and self.app._do_add_to_library(dlg.result):
            self.app._set_status(f"Added '{dlg.result['name']}' to library.")
            self._render()

    def _add_one(self, name):
        """The row 'Add' button: start transcribing this library feed."""
        entry = self.app._lib_find(name)
        if entry and self.app._do_add(dict(entry)):
            self.app._set_status(f"Added '{name}' to transcription.")
        self._render()

    def _remove_one(self, name):
        """The row 'Remove' button: stop transcribing (stays in the library)."""
        self.app._do_remove([name])
        self.app._set_status(f"Removed '{name}' from transcription.")
        self._render()

    def _edit(self, name):
        entry = self.app._lib_find(name)
        if not entry:
            return
        dlg = AddStreamDialog(self, title=f"Edit '{name}'", initial=entry)
        if dlg.result:
            self.app._do_edit(name, dlg.result)
            self._render()

    def _delete(self, name):
        if messagebox.askyesno("Delete feed",
                               f"Permanently remove “{name}” from the library?",
                               parent=self):
            self.app._lib_delete([name])
            self._render()


class TTSDialog(tk.Toplevel):
    """Text-to-speech settings: read selected feeds and/or keyword-matching lines
    aloud in a clear neural voice. Persists to config via app._save_tts_cfg."""
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.cfg = app.tts_cfg
        self.title("Read aloud (text-to-speech)")
        self.geometry("480x768")
        self.minsize(440, 400)
        self.configure(bg=BG)
        self.transient(parent)

        if not core.tts_available():
            tk.Label(self, text="Text-to-speech isn't available.\nNo Piper voice "
                     "model found in tts_voices/.", bg=BG, fg=FG, justify="left",
                     font=("Segoe UI", 10)).pack(padx=16, pady=20)
            tk.Button(self, text="Close", command=self.destroy, bg=BG2, fg=FG,
                      relief="flat").pack(pady=8)
            return

        # Button bar pinned to the bottom FIRST so it's always visible, then a
        # scrollable area for the (now tall) settings content above it.
        btns = tk.Frame(self, bg=BG2); btns.pack(side="bottom", fill="x")
        tk.Button(btns, text="🔊 Test voice", command=self._test,
                  bg=BG2, fg=FG, relief="flat").pack(side="left", padx=4, pady=6)
        tk.Button(btns, text="Save", command=self._save,
                  bg=BG2, fg=FG, relief="flat").pack(side="right", padx=4, pady=6)
        tk.Button(btns, text="Close", command=self.destroy,
                  bg=BG2, fg=FG, relief="flat").pack(side="right", padx=4, pady=6)

        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        body = tk.Frame(canvas, bg=BG)
        win = canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        # Mouse-wheel scrolling only while the pointer is over this dialog
        # (bound on Enter, released on Leave, so it never affects other windows).
        def _wheel(e):
            canvas.yview_scroll(int(-e.delta / 120), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _wheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

        pad = {"padx": 12, "anchor": "w"}
        # Enable toggle.
        row = tk.Frame(body, bg=BG); row.pack(fill="x", pady=(12, 4), **pad)
        self.enabled = CheckBox(row, value=self.cfg.get("enabled", False))
        self.enabled.pack(side="left")
        tk.Label(row, text="  Read transcriptions aloud", bg=BG, fg=FG,
                 font=("Segoe UI", 10, "bold")).pack(side="left")

        # Voice picker.
        vrow = tk.Frame(body, bg=BG); vrow.pack(fill="x", pady=4, **pad)
        tk.Label(vrow, text="Voice:", bg=BG, fg=MUTED).pack(side="left")
        voices = [v[0] for v in core.list_tts_voices()]
        self.voice_var = tk.StringVar(value=self.cfg.get("voice") or (voices[0] if voices else ""))
        ttk.Combobox(vrow, textvariable=self.voice_var, values=voices,
                     state="readonly", width=28).pack(side="left", padx=6)

        # What to read: mode.
        tk.Label(body, text="What to read:", bg=BG, fg=MUTED).pack(pady=(10, 2), **pad)
        self.mode = tk.StringVar(value=self.cfg.get("mode", "feeds"))
        for val, lab in [("feeds", "Selected feeds"),
                         ("keywords", "Only lines with my keywords"),
                         ("both", "Selected feeds + keyword matches")]:
            mrow = tk.Frame(body, bg=BG); mrow.pack(fill="x", **pad)
            tk.Radiobutton(mrow, text=lab, variable=self.mode, value=val, bg=BG,
                           fg=FG, selectcolor=BG2, activebackground=BG,
                           activeforeground=FG).pack(side="left")

        # Feed multi-select.
        tk.Label(body, text="Feeds to read (only active feeds shown):",
                 bg=BG, fg=MUTED).pack(pady=(10, 2), **pad)
        fbox = tk.Frame(body, bg=BG); fbox.pack(fill="x", **pad)
        self.feed_checks = {}
        selected = set(self.cfg.get("feeds", []))
        # Only feeds currently being transcribed (enabled) are selectable here.
        names = [s["name"] for s in self.app.streams if core.is_enabled(s)]
        # Preserve any saved selections for feeds that aren't active right now, so
        # saving from this dialog doesn't silently drop them.
        self._inactive_selected = [n for n in selected if n not in names]
        if not names:
            tk.Label(fbox, text="(no feeds active — start one from Feeds)",
                     bg=BG, fg=MUTED).pack(anchor="w")
        for nm in names:
            frow = tk.Frame(fbox, bg=BG); frow.pack(fill="x", anchor="w")
            cb = CheckBox(frow, value=(nm in selected))
            cb.pack(side="left")
            tk.Label(frow, text=f"  {nm}", bg=BG, fg=FG).pack(side="left")
            self.feed_checks[nm] = cb

        # Keyword presets (checkboxes, 2 columns) + free-text extras.
        khdr = tk.Frame(body, bg=BG); khdr.pack(fill="x", pady=(10, 2), **pad)
        tk.Label(khdr, text="Alert keywords (check presets):", bg=BG,
                 fg=MUTED).pack(side="left")
        tk.Button(khdr, text="Check all", command=lambda: self._set_all_presets(True),
                  bg=BG2, fg=FG, relief="flat").pack(side="left", padx=(10, 2))
        tk.Button(khdr, text="Clear", command=lambda: self._set_all_presets(False),
                  bg=BG2, fg=FG, relief="flat").pack(side="left", padx=2)
        pbox = tk.Frame(body, bg=BG); pbox.pack(fill="x", padx=12, anchor="w")
        self.preset_checks = {}
        checked_presets = set(self.cfg.get("keyword_presets", []))
        for i, (label, _terms) in enumerate(KEYWORD_PRESETS):
            cell = tk.Frame(pbox, bg=BG)
            cell.grid(row=i // 2, column=i % 2, sticky="w", padx=(0, 14))
            cb = CheckBox(cell, value=(label in checked_presets))
            cb.pack(side="left")
            tk.Label(cell, text=f"  {label}", bg=BG, fg=FG).pack(side="left")
            self.preset_checks[label] = cb

        tk.Label(body, text="Extra keywords (comma-separated):", bg=BG, fg=MUTED).pack(
            pady=(10, 2), **pad)
        self.kw_var = tk.StringVar(value=", ".join(self.cfg.get("keywords", [])))
        tk.Entry(body, textvariable=self.kw_var, width=44).pack(**pad)
        tk.Frame(body, bg=BG, height=8).pack()   # bottom breathing room

    def _set_all_presets(self, value):
        for cb in self.preset_checks.values():
            cb.set(value)

    def _collect(self):
        # Checked active feeds + any saved selections for feeds not currently
        # active (so switching a feed off doesn't wipe its TTS selection).
        feeds = [n for n, cb in self.feed_checks.items() if cb.get()]
        feeds += getattr(self, "_inactive_selected", [])
        presets = [lab for lab, cb in self.preset_checks.items() if cb.get()]
        extras = [k.strip() for k in self.kw_var.get().split(",") if k.strip()]
        return {
            "enabled": self.enabled.get(),
            "voice": self.voice_var.get() or None,
            "mode": self.mode.get(),
            "feeds": feeds,
            "keyword_presets": presets,       # checked preset labels
            "keywords": extras,               # free-text extras
        }

    def _save(self):
        self.app.tts_cfg = self._collect()
        self.cfg = self.app.tts_cfg
        self.app._save_tts_cfg()
        self.app._set_status("TTS settings saved.")
        self.destroy()

    def _test(self):
        # Apply current settings enough to test, then speak a sample line.
        self.app.tts_cfg = self._collect()
        self.app._save_tts_cfg()
        if self.app.engine:
            self.app.engine.tts_speak_test(
                "Dispatch, be advised, this is a text to speech test.")


HELP_TEXT = """\
SCANNER TRANSCRIBER — USER GUIDE

WHAT IT DOES
  Listens to police/emergency scanner feeds (and audio playing on your PC) and
  transcribes them live on your GPU, so you can read what's said instead of
  straining to hear garbled radio. Multiple feeds run at once, each in its own
  color-coded column.

GETTING STARTED
  • On launch, the splash shows while the speech model loads, then the main
    window appears with your saved feeds already running.
  • Nothing plays through your speakers unless you choose to listen — the app
    reads the audio directly.

FEEDS  (toolbar "Feeds" button / Streams menu)
  A single window manages every feed you've saved (your "library").
  • Add      – start transcribing a saved feed.
  • Remove   – stop transcribing it (it stays saved to re-add later).
  • Edit     – fix a typo, rename, change color, or update a URL. If the feed is
               active it restarts with the new settings.
  • Delete   – forget a feed permanently.
  • + Add new feed – save a new feed (Broadcastify feed id / URL, or a PC-audio
                     or application source). New feeds are saved but NOT started
                     until you click Add.
  • Drag the ⠿ handle to reorder feeds; the order is remembered.

VIEWS  (toolbar / View menu)
  • Sectors  – one scrolling column per feed, side by side. Drag a column's
               header (⠿) to rearrange; right-click a column for options.
  • Unified  – one combined feed, each line tagged with which feed it came from.
  • Color by – "stream" colors by feed; "unit" colors by the spoken call sign
               (e.g. Adam 33) and makes call signs clickable to filter to just
               that unit. "Show all units" clears the filter.
  • Font +/- buttons (or Ctrl +/- , Ctrl 0) resize the transcript text.

LISTEN  ("Listen to:" dropdown)
  Hear one feed through your speakers while the rest keep transcribing silently.
  Choose (none) to stay muted.

PC AUDIO & APPLICATIONS
  Add custom stream → Source:
  • "pc audio"    – transcribe everything from a chosen speaker/output device.
  • "application" – transcribe ONE running app's audio (pick it from the list;
                    click ↻ to refresh). Great for a TV stream or media player.
  Tip: audio must actually be playing through the chosen output/app. (These work
  only at the physical PC, not over Remote Desktop.)

READ ALOUD — TEXT TO SPEECH  (toolbar "Speak" button)
  Have transcriptions read to you in a clear neural voice.
  • Turn it on, pick a voice, and choose what to read:
      – Selected feeds, or
      – Only lines containing your keywords, or
      – Both.
  • Feeds to read: check any active feed.
  • Alert keywords: check preset categories (Shooting, Fire, Pursuit, …) — each
    expands to several synonyms automatically. Use "Check all" / "Clear", and add
    your own comma-separated terms in "Extra keywords".
  • The line being read is highlighted in the transcript so you can follow along.
  • "Test voice" speaks a sample.

MODEL & UPDATES
  • Model dropdown switches the Whisper model live (bigger = more accurate,
    smaller = faster/lighter). It reloads without stopping your feeds.
  • "Check updates" (toolbar) looks for newer speech-engine libraries (report
    only; it never installs on its own). It also checks quietly at launch.
  • Help → "Check for app updates" looks for a newer Transcriber release on
    GitHub. If one exists it shows the release notes and can download the
    installer and upgrade in place (you'll get a Windows admin prompt). The app
    also checks for a new version quietly at launch.

LOGS & PRIVACY
  Transcripts are saved under logs\\ and auto-deleted after a number of days
  (log_retention_days in config.json). Feeds may contain sensitive info — keep
  transcripts to yourself.

NOTES
  • "Thank you"-type phantom lines are Whisper hallucinating on silence; the app
    filters the obvious ones out.
  • Encrypted channels can't be transcribed (there's nothing in the clear to
    hear) — this only works on unencrypted feeds.
"""


class HelpDialog(tk.Toplevel):
    """Scrollable user guide covering all features."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Help — Scanner Transcriber")
        self.geometry("640x680")
        self.minsize(480, 400)
        self.configure(bg=BG)
        self.transient(parent)

        btns = tk.Frame(self, bg=BG2); btns.pack(side="bottom", fill="x")
        tk.Button(btns, text="Close", command=self.destroy, bg=BG2, fg=FG,
                  relief="flat").pack(side="right", padx=6, pady=6)

        frame = tk.Frame(self, bg=BG); frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, bg="#16171a", fg=FG, wrap="word", relief="flat",
                      font=("Segoe UI", 10), padx=14, pady=12)
        sb = ttk.Scrollbar(frame, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.insert("1.0", HELP_TEXT)
        txt.configure(state="disabled")


class AboutDialog(tk.Toplevel):
    """About page: developer photo + app credit."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("About")
        self.configure(bg=BG)
        self.transient(parent)
        self.resizable(False, False)

        # Developer photo, high-quality scaled to ~420px tall (LANCZOS via Pillow).
        if os.path.exists(DEVELOPER_PHOTO):
            img = load_scaled_image(DEVELOPER_PHOTO, max_h=420)
            if img is not None:
                self._img = img
                tk.Label(self, image=img, bg=BG, borderwidth=0).pack(
                    padx=20, pady=(20, 10))

        tk.Label(self, text="Transcriber", bg=BG, fg=FG,
                 font=("Segoe UI", 16, "bold")).pack()
        tk.Label(self, text="by DevCon Productions", bg=BG, fg=FG,
                 font=("Segoe UI", 11)).pack(pady=(2, 0))
        tk.Label(self, text="Cleveland, Ohio, United States", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack()
        tk.Label(self, text=f"Version {APP_VERSION}", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(pady=(6, 0))
        tk.Label(self, text="© 2026 DevCon Productions", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(pady=(2, 0))

        tk.Button(self, text="Close", command=self.destroy, bg=BG2, fg=FG,
                  relief="flat").pack(pady=16)

        # Center over the parent.
        self.update_idletasks()
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass


class BroadcastifyLoginDialog(tk.Toplevel):
    """Enter / update Broadcastify Premium credentials. Prefilled with the
    current login; saving persists to credentials.json and (via on_save) applies
    it live so running feeds reconnect without an app restart."""
    def __init__(self, parent, username="", password="", on_save=None):
        super().__init__(parent)
        self.title("Broadcastify login")
        self.configure(bg=BG)
        self.transient(parent)
        self.resizable(False, False)
        self._on_save = on_save

        tk.Label(self, text="Broadcastify Premium account", bg=BG, fg=FG,
                 font=("Segoe UI", 13, "bold")).pack(padx=26, pady=(20, 2))
        tk.Label(self, text="Required to stream Broadcastify feeds. Stored only on\n"
                            "this PC in credentials.json — never uploaded.",
                 bg=BG, fg=MUTED, font=("Segoe UI", 9), justify="center").pack(padx=26)

        form = tk.Frame(self, bg=BG)
        form.pack(padx=26, pady=(16, 4), fill="x")
        tk.Label(form, text="Username", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).grid(row=0, column=0, sticky="w", pady=5)
        self.user_var = tk.StringVar(value=username or "")
        u = tk.Entry(form, textvariable=self.user_var, width=26, bg=BG2, fg=FG,
                     insertbackground=FG, relief="flat")
        u.grid(row=0, column=1, padx=(12, 0), pady=5, ipady=2)

        tk.Label(form, text="Password", bg=BG, fg=FG,
                 font=("Segoe UI", 10)).grid(row=1, column=0, sticky="w", pady=5)
        self.pw_var = tk.StringVar(value=password or "")
        self.pw_entry = tk.Entry(form, textvariable=self.pw_var, width=26, show="•",
                                 bg=BG2, fg=FG, insertbackground=FG, relief="flat")
        self.pw_entry.grid(row=1, column=1, padx=(12, 0), pady=5, ipady=2)

        show = tk.Frame(self, bg=BG)
        show.pack(padx=26, anchor="e")
        self._show = CheckBox(show, value=False, command=self._toggle_pw,
                              font=("Segoe UI", 11))
        self._show.pack(side="left")
        tk.Label(show, text="Show password", bg=BG, fg=MUTED,
                 font=("Segoe UI", 9)).pack(side="left", padx=(4, 0))

        self.msg = tk.Label(self, text="", bg=BG, fg="#e06c6c",
                            font=("Segoe UI", 9))
        self.msg.pack(pady=(6, 0))

        btns = tk.Frame(self, bg=BG)
        btns.pack(pady=(8, 18))
        tk.Button(btns, text="Save", command=self._save, bg=BG2, fg=FG,
                  relief="flat", width=10).pack(side="left", padx=6)
        tk.Button(btns, text="Cancel", command=self.destroy, bg=BG2, fg=FG,
                  relief="flat", width=10).pack(side="left", padx=6)

        (u if not username else self.pw_entry).focus_set()
        self.bind("<Return>", lambda e: self._save())
        self.bind("<Escape>", lambda e: self.destroy())
        self.update_idletasks()
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass

    def _toggle_pw(self):
        self.pw_entry.configure(show="" if self._show.get() else "•")

    def _save(self):
        user = self.user_var.get().strip()
        pw = self.pw_var.get().strip()
        if not user or not pw:
            self.msg.configure(text="Enter both a username and password.")
            return
        if self._on_save:
            self._on_save(user, pw)
        self.destroy()


class AppUpdateDialog(tk.Toplevel):
    """Offer a newer app version: show the version + release notes, then a
    Download & install button that swaps to a progress bar. The controller pushes
    progress via set_progress()/on_done()/on_error()."""
    def __init__(self, parent, info, on_download, on_view):
        super().__init__(parent)
        self.title("Update available")
        self.configure(bg=BG)
        self.transient(parent)
        self.resizable(False, False)
        self._on_download = on_download
        self._on_view = on_view
        self._info = info

        tk.Label(self, text=f"Transcriber {info['latest']} is available",
                 bg=BG, fg=FG, font=("Segoe UI", 13, "bold")).pack(
            padx=24, pady=(20, 2))
        tk.Label(self, text=f"You have {info['current']}.", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(padx=24)

        # Release notes (read-only, scrollable).
        nf = tk.Frame(self, bg=BG)
        nf.pack(padx=24, pady=(14, 6), fill="both")
        sb = ttk.Scrollbar(nf)
        sb.pack(side="right", fill="y")
        txt = tk.Text(nf, width=64, height=12, wrap="word", bg=BG2, fg=FG,
                      relief="flat", font=("Segoe UI", 9), yscrollcommand=sb.set,
                      padx=10, pady=8)
        txt.pack(side="left", fill="both", expand=True)
        sb.config(command=txt.yview)
        txt.insert("1.0", info.get("notes") or "(no release notes)")
        txt.configure(state="disabled")

        # Progress area (hidden until download starts).
        self._pbar = ttk.Progressbar(self, orient="horizontal", mode="determinate",
                                     maximum=100, length=460)
        self._status = tk.Label(self, text="", bg=BG, fg=MUTED,
                                font=("Segoe UI", 9))

        self._btns = tk.Frame(self, bg=BG)
        self._btns.pack(pady=(6, 18))
        self._dl_btn = tk.Button(self._btns, text="Download & install",
                                 command=self._start, bg=BG2, fg=FG, relief="flat",
                                 width=18)
        self._dl_btn.pack(side="left", padx=6)
        tk.Button(self._btns, text="View release", command=self._on_view, bg=BG2,
                  fg=FG, relief="flat", width=12).pack(side="left", padx=6)
        self._later_btn = tk.Button(self._btns, text="Later", command=self.destroy,
                                    bg=BG2, fg=FG, relief="flat", width=10)
        self._later_btn.pack(side="left", padx=6)

        self.update_idletasks()
        try:
            px = parent.winfo_rootx() + (parent.winfo_width() - self.winfo_width()) // 2
            py = parent.winfo_rooty() + (parent.winfo_height() - self.winfo_height()) // 2
            self.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass

    def _start(self):
        self._dl_btn.configure(state="disabled")
        self._later_btn.configure(state="disabled")
        self._pbar.pack(padx=24, pady=(0, 2))
        self._status.pack(padx=24, pady=(0, 4))
        self._status.configure(text="Starting download…")
        self._on_download()

    def set_progress(self, done, total):
        if total:
            pct = int(done * 100 / total)
            self._pbar.configure(mode="determinate", value=pct)
            self._status.configure(
                text=f"Downloading… {done // (1 << 20)} / {total // (1 << 20)} MB ({pct}%)")
        else:
            self._pbar.configure(mode="indeterminate")
            self._pbar.start(20)
            self._status.configure(text=f"Downloading… {done // (1 << 20)} MB")

    def on_done(self):
        try:
            self._pbar.stop()
        except Exception:
            pass
        self._pbar.configure(mode="determinate", value=100)
        self._status.configure(text="Download complete — launching installer…",
                               fg="#7fd18b")

    def on_error(self, msg):
        try:
            self._pbar.stop()
        except Exception:
            pass
        self._status.configure(text=f"Update failed: {msg}", fg="#e06c6c")
        self._dl_btn.configure(state="normal")
        self._later_btn.configure(state="normal")


class TranscriberGUI:
    def __init__(self, root, splash=None):
        self.root = root
        self._splash = splash
        self._splash_min_deadline = None   # earliest time we may close the splash
        self.root.title("Transcriber by DevCon Productions")
        self.root.geometry("1100x680")
        self.root.configure(bg=BG)
        self._set_window_icon()
        if splash is not None:
            import time as _t
            self._splash_min_deadline = _t.monotonic() + 2.0   # 2s minimum

        self.cfg = self._load_cfg()
        self.engine = None
        self.events = queue.Queue()          # (kind, payload) from bg threads
        self.sector_panels = {}              # name -> ScrolledText (sectors view)
        self._update_dialog = None           # active AppUpdateDialog, if any
        self.history = collections.deque(maxlen=HISTORY_MAX)  # (name,color,text,ts)
        self.view_mode = tk.StringVar(value=self.cfg.get("view_mode", "sectors"))
        self.color_mode = tk.StringVar(value="stream")   # "stream" | "unit"
        self.model_var = tk.StringVar(value=self.cfg.get("model", "large-v3"))
        self.listen_var = tk.StringVar(value="(none)")
        self.streams = list(self.cfg.get("streams", []))
        self.library = self._seed_library()   # persistent catalog of saved feeds
        self.extra_prefixes = self.cfg.get("unit_prefixes", [])
        self._unit_colors = {}                            # unit -> hex (stable)
        self.filter_unit = None                           # None = show all units
        self.tts_cfg = dict(self.cfg.get("tts", {}))      # TTS settings (persisted)

        # Shared transcript font: all panels use this one object, so resizing it
        # updates every panel live (no rebuild). Size persists in config.json.
        size = int(self.cfg.get("font_size", FONT_DEFAULT))
        size = max(FONT_MIN, min(FONT_MAX, size))
        self.font_size = tk.IntVar(value=size)
        self.transcript_font = tkfont.Font(family=FONT_FAMILY, size=size)

        self._build_menu()
        self._build_toolbar()
        self._build_body()
        self._set_status("Loading model...")

        # Keyboard shortcuts for font size (Ctrl +/-/0), like a browser.
        self.root.bind("<Control-plus>", lambda e: self._change_font(+1))
        self.root.bind("<Control-equal>", lambda e: self._change_font(+1))
        self.root.bind("<Control-KP_Add>", lambda e: self._change_font(+1))
        self.root.bind("<Control-minus>", lambda e: self._change_font(-1))
        self.root.bind("<Control-KP_Subtract>", lambda e: self._change_font(-1))
        self.root.bind("<Control-0>", lambda e: self._change_font(0))

        # Bring the window to the front on launch (it otherwise opens behind
        # whatever has focus). Pin topmost briefly, then release so it doesn't
        # stay stuck above other windows.
        self._raise_window()

        # Start engine on a background thread so the window paints immediately.
        threading.Thread(target=self._start_engine, daemon=True).start()
        self.root.after(100, self._drain_events)
        # Quiet automatic update check shortly after launch (status bar only).
        self.root.after(1500, self._check_updates_auto)
        self.root.after(3500, self._check_app_update_auto)   # quiet app-update check
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Safety net: never leave the splash up (and the window hidden) forever
        # if the model fails to load and no 'ready' event fires.
        if self._splash is not None:
            self.root.after(120000, self._dismiss_splash)

    def _raise_window(self):
        try:
            self.root.lift()
            self.root.attributes("-topmost", True)
            self.root.focus_force()
            self.root.after(400, lambda: self.root.attributes("-topmost", False))
        except Exception:
            pass

    def _set_window_icon(self):
        """Set the title-bar + taskbar icon from the icon PNG (if present).
        Keeps a reference so the image isn't garbage-collected."""
        if not os.path.exists(TASKBAR_ICON):
            return
        try:
            self._icon_img = tk.PhotoImage(file=TASKBAR_ICON)
            self.root.iconphoto(True, self._icon_img)
        except Exception:
            pass

    def _dismiss_splash(self):
        """Close the splash and reveal the main window. Enforces the 2s minimum:
        if we hit 'ready' sooner, reschedule until the minimum has elapsed."""
        if not self._splash:
            return
        import time as _t
        remaining = (self._splash_min_deadline or 0) - _t.monotonic()
        if remaining > 0:
            self.root.after(int(remaining * 1000) + 20, self._dismiss_splash)
            return
        try:
            self._splash.destroy()
        except Exception:
            pass
        self._splash = None
        try:
            self.root.deiconify()
            self._raise_window()
        except Exception:
            pass

    # ----- config ----------------------------------------------------------
    def _load_cfg(self):
        try:
            return core.load_config()
        except Exception:
            return {"model": "large-v3", "device": "cuda", "compute_type": "float16",
                    "language": "en", "beam_size": 5, "vad": {}, "filters": {},
                    "streams": []}

    def _seed_library(self):
        """Build the persistent feed library: the built-in catalog, plus any feeds
        saved from prior sessions (config 'feed_library'), plus any currently-active
        streams not already listed. Keyed/de-duped by name; first occurrence wins."""
        lib, seen = [], set()
        def add(entry):
            n = entry.get("name")
            if n and n not in seen:
                seen.add(n)
                lib.append(dict(entry))
        for e in self.cfg.get("feed_library", []):   # user's saved library (persisted)
            add(e)
        for e in FEED_CATALOG:                        # built-in defaults
            add(e)
        for s in self.cfg.get("streams", []):         # active feeds not yet in library
            add({k: v for k, v in s.items() if k != "disabled"})
        return lib

    def _lib_find(self, name):
        for e in self.library:
            if e["name"] == name:
                return e
        return None

    def _lib_upsert(self, entry):
        """Insert or update a library entry by name (without the 'disabled' flag)."""
        clean = {k: v for k, v in entry.items() if k != "disabled"}
        existing = self._lib_find(clean["name"])
        if existing:
            existing.clear()
            existing.update(clean)
        else:
            self.library.append(clean)

    def _do_add_to_library(self, entry):
        """Add a feed to the LIBRARY ONLY (does not start transcribing it).
        Returns True on success, False if the name already exists in the library."""
        if self._lib_find(entry["name"]):
            messagebox.showwarning("Duplicate",
                                   f"A feed named '{entry['name']}' is already in the library.")
            return False
        self._lib_upsert(entry)
        self._save_cfg()
        return True

    def _reorder_library(self, name, target_name):
        """Move library feed `name` to the position of `target_name`, in-list.
        Direction-aware: dragging DOWN lands after the target, UP lands before,
        so the row ends up where dropped. Persisted to config (survives sessions)."""
        names = [e["name"] for e in self.library]
        if name == target_name or name not in names or target_name not in names:
            return
        src = names.index(name)
        dragging_down = src < names.index(target_name)
        moving = self.library.pop(src)
        t = next(i for i, e in enumerate(self.library) if e["name"] == target_name)
        self.library.insert(t + 1 if dragging_down else t, moving)
        self._save_cfg()

    def _save_cfg(self):
        self.cfg["streams"] = self.streams
        self.cfg["feed_library"] = self.library
        self.cfg["view_mode"] = self.view_mode.get()
        try:
            with open(core.CONFIG_PATH, "w", encoding="utf-8") as f:
                import json
                json.dump(self.cfg, f, indent=2)
            self._set_status("Saved config.json")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # ----- UI construction -------------------------------------------------
    def _build_menu(self):
        m = tk.Menu(self.root)
        streams_menu = tk.Menu(m, tearoff=0)
        streams_menu.add_command(label="Feeds...", command=self._open_library)
        streams_menu.add_command(label="Broadcastify login...", command=self._open_login)
        streams_menu.add_separator()
        streams_menu.add_command(label="Save config", command=self._save_cfg)
        m.add_cascade(label="Streams", menu=streams_menu)

        view_menu = tk.Menu(m, tearoff=0)
        view_menu.add_radiobutton(label="Unified feed", variable=self.view_mode,
                                  value="unified", command=self._rebuild_body)
        view_menu.add_radiobutton(label="Sectors (split)", variable=self.view_mode,
                                  value="sectors", command=self._rebuild_body)
        view_menu.add_separator()
        view_menu.add_command(label="Clear", command=self._clear_text)
        m.add_cascade(label="View", menu=view_menu)

        help_menu = tk.Menu(m, tearoff=0)
        help_menu.add_command(label="Help / User guide", command=self._open_help)
        help_menu.add_command(label="Check for app updates...",
                              command=self._check_app_update_manual)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._open_about)
        m.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=m)

    def _open_help(self):
        HelpDialog(self.root)

    def _open_about(self):
        AboutDialog(self.root)

    # -- app self-update ----------------------------------------------------
    def _check_app_update_manual(self):
        self._set_status("Checking for app updates…")
        threading.Thread(target=self._run_app_update_check, args=(True,),
                         daemon=True).start()

    def _check_app_update_auto(self):
        threading.Thread(target=self._run_app_update_check, args=(False,),
                         daemon=True).start()

    def _run_app_update_check(self, manual):
        info = core.check_for_app_update(APP_VERSION)
        self.events.put(("app_update_result", (info, manual)))

    def _handle_app_update_result(self, info, manual):
        if info is None:
            if manual:
                messagebox.showinfo("Updates",
                                    "Couldn't check for updates (offline or GitHub "
                                    "unavailable). Try again later.")
            else:
                self._set_status("Update check skipped (offline).")
            return
        if not info.get("available"):
            if manual:
                messagebox.showinfo("Updates",
                                    f"You're up to date (version {info['current']}).")
            else:
                self._set_status(f"Up to date (v{info['current']}).")
            return
        if not info.get("asset_url"):
            # A newer version exists but has no installer asset to download.
            self._set_status(f"Version {info['latest']} is available on GitHub.")
            if manual:
                messagebox.showinfo("Update available",
                                    f"Transcriber {info['latest']} is available, but "
                                    "no installer was attached to the release. See "
                                    "the releases page.")
            return
        self._set_status(f"Update available: v{info['latest']}.")
        self._update_dialog = AppUpdateDialog(
            self.root, info,
            on_download=lambda: self._start_app_download(info),
            on_view=lambda: self._open_url(info.get("html_url")))

    def _start_app_download(self, info):
        dest = os.path.join(core.DATA_DIR, "updates", info["asset_name"])
        threading.Thread(target=self._download_update, args=(info, dest),
                         daemon=True).start()

    def _download_update(self, info, dest):
        try:
            core.download_file(
                info["asset_url"], dest,
                progress_cb=lambda d, t: self.events.put(("app_dl_progress", (d, t))))
            self.events.put(("app_dl_done", dest))
        except Exception as e:
            self.events.put(("app_dl_error", str(e)))

    def _launch_installer_and_quit(self, path):
        """Launch the downloaded installer (UAC prompt — it's admin-manifested)
        and quit so it can replace files. Inno upgrades in place via its fixed
        AppId and relaunches the app from its postinstall step."""
        try:
            os.startfile(path)   # noqa: S606 — trusted, freshly downloaded from our release
        except Exception as e:
            if getattr(self, "_update_dialog", None):
                self._update_dialog.on_error(f"couldn't launch installer: {e}")
            return
        # Give the installer a moment to spawn before we exit.
        self.root.after(1200, self._on_close)

    def _open_url(self, url):
        if not url:
            return
        import webbrowser
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _open_login(self):
        """Open the Broadcastify credentials dialog; apply saved creds live."""
        user, pw = core.load_credentials()

        def on_save(u, p):
            if self.engine is not None:
                self.engine.apply_credentials(u, p, self.streams)
            else:
                core.save_credentials(u, p)
            self._set_status(f"Broadcastify login saved (user '{u}'). "
                             "Feeds will reconnect.")
        BroadcastifyLoginDialog(self.root, username=user or "", password=pw or "",
                                on_save=on_save)

    def _maybe_prompt_login(self):
        """On startup, if Broadcastify feeds are configured but no usable login
        is set, open the login dialog so feeds don't silently drop."""
        try:
            needs = any(core.is_broadcastify_stream(s)
                        for s in self._enabled_streams())
            if needs and not core.credentials_configured():
                self._set_status("Broadcastify feeds need a login to stream — "
                                 "enter your account details.")
                self._open_login()
        except Exception:
            pass

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=BG2)
        bar.pack(side="top", fill="x")

        tk.Button(bar, text="Feeds", command=self._open_library,
                  bg=BG2, fg=FG, relief="flat").pack(side="left", padx=4, pady=4)
        tk.Button(bar, text="Speak", command=self._open_tts,
                  bg=BG2, fg=FG, relief="flat").pack(side="left", padx=4, pady=4)

        tk.Label(bar, text="  View:", bg=BG2, fg=MUTED).pack(side="left")
        ttk.Combobox(bar, textvariable=self.view_mode, values=["unified", "sectors"],
                     state="readonly", width=10).pack(side="left", padx=4)
        self.view_mode.trace_add("write", lambda *_: self._rebuild_body())

        tk.Label(bar, text="  Color by:", bg=BG2, fg=MUTED).pack(side="left")
        ttk.Combobox(bar, textvariable=self.color_mode, values=["stream", "unit"],
                     state="readonly", width=8).pack(side="left", padx=4)
        self.color_mode.trace_add("write", lambda *_: self._on_color_mode_change())

        tk.Label(bar, text="  Listen to:", bg=BG2, fg=MUTED).pack(side="left")
        self.listen_combo = ttk.Combobox(bar, textvariable=self.listen_var,
                                          values=["(none)"], state="readonly", width=18)
        self.listen_combo.pack(side="left", padx=4)
        self.listen_combo.bind("<<ComboboxSelected>>", self._on_listen_change)

        # Font-size controls (right side of the first row).
        tk.Button(bar, text="A+", command=lambda: self._change_font(+1),
                  bg=BG2, fg=FG, relief="flat", width=3).pack(side="right", padx=2, pady=4)
        self.font_label = tk.Label(bar, text=str(self.font_size.get()),
                                   bg=BG2, fg=MUTED, width=2)
        self.font_label.pack(side="right")
        tk.Button(bar, text="A−", command=lambda: self._change_font(-1),
                  bg=BG2, fg=FG, relief="flat", width=3).pack(side="right", padx=2, pady=4)
        tk.Label(bar, text="  Font:", bg=BG2, fg=MUTED).pack(side="right")

        # Second row: unit filter (set by clicking a call sign in unit-color mode)
        # on the left, and the Whisper model picker on the right.
        bar2 = tk.Frame(self.root, bg=BG2)
        bar2.pack(side="top", fill="x")
        self.filter_label = tk.Label(bar2, text="  Filter: (all units)", bg=BG2, fg=MUTED)
        self.filter_label.pack(side="left")
        self.filter_clear_btn = tk.Button(bar2, text="Show all units",
                                          command=self.clear_unit_filter,
                                          bg=BG2, fg=FG, relief="flat", state="disabled")
        self.filter_clear_btn.pack(side="left", padx=6, pady=3)
        tk.Label(bar2, text="(tip: set 'Color by: unit', then click a call sign to "
                            "follow just that unit)", bg=BG2, fg=MUTED).pack(side="left", padx=8)

        # Model picker (right side). Switching reloads Whisper live.
        self.model_combo = ttk.Combobox(bar2, textvariable=self.model_var,
                                        values=MODEL_CHOICES, state="readonly", width=16)
        self.model_combo.pack(side="right", padx=6, pady=3)
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_change)
        tk.Label(bar2, text="Model:", bg=BG2, fg=MUTED).pack(side="right")

        self.update_btn = tk.Button(bar2, text="Check updates",
                                    command=self._check_updates_manual,
                                    bg=BG2, fg=FG, relief="flat")
        self.update_btn.pack(side="right", padx=6, pady=3)

    def _build_body(self):
        self.body = tk.Frame(self.root, bg=BG)
        self.body.pack(side="top", fill="both", expand=True)

        self.status = tk.Label(self.root, text="", bg=BG2, fg=MUTED, anchor="w")
        self.status.pack(side="bottom", fill="x")

        self._build_view()

    def _build_view(self):
        for w in self.body.winfo_children():
            w.destroy()
        self.sector_panels.clear()

        active = self._enabled_streams()
        if self.view_mode.get() == "unified":
            self.unified = self._make_text(self.body)
            self.unified.pack(fill="both", expand=True, padx=6, pady=6)
            # Color tags are configured lazily per line in _render_line.
        else:
            panels = active or [{"name": "(no active streams)", "color": "grey"}]
            self._sector_headers = []   # (frame, stream_name, hdr) for drag hit-test
            for i, s in enumerate(panels):
                col = tk.Frame(self.body, bg=BG)
                col.grid(row=0, column=i, sticky="nsew", padx=3, pady=3)
                self.body.grid_columnconfigure(i, weight=1)
                self.body.grid_rowconfigure(0, weight=1)
                hdr = tk.Label(col, text=("⠿ " + s["name"]), bg=BG2,
                               fg=COLOR_HEX.get(s.get("color", "white"), FG),
                               font=("Segoe UI", 10, "bold"), cursor="fleur")
                hdr.pack(side="top", fill="x")
                txt = self._make_text(col)
                txt.pack(fill="both", expand=True)
                if active:
                    self.sector_panels[s["name"]] = txt
                    self._sector_headers.append((col, s["name"], hdr))
                    # Drag the header to reorder columns.
                    hdr.bind("<ButtonPress-1>",
                             lambda e, n=s["name"], c=s.get("color", "white"):
                             self._drag_start(n, c, e))
                    hdr.bind("<B1-Motion>", self._drag_motion)
                    hdr.bind("<ButtonRelease-1>", self._drag_drop)
                    # Right-click for the sector context menu.
                    hdr.bind("<Button-3>", lambda e, n=s["name"]: self._sector_menu(e, n))
                    txt.bind("<Button-3>", lambda e, n=s["name"]: self._sector_menu(e, n))

        self._replay_history()

    def _make_text(self, parent):
        frame = tk.Frame(parent, bg=BG)
        txt = tk.Text(frame, bg="#16171a", fg=FG, insertbackground=FG,
                      wrap="word", relief="flat", font=self.transcript_font,
                      state="disabled")
        sb = ttk.Scrollbar(frame, command=txt.yview)
        txt.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        # Return a thin wrapper that .pack/.grid forwards to the frame.
        txt.frame = frame
        txt.pack = frame.pack
        txt.grid = frame.grid
        return txt

    def _rebuild_body(self):
        self._build_view()
        self._refresh_listen_choices()
        # Persist the view choice if it changed (so it's the default next launch).
        if self.cfg.get("view_mode") != self.view_mode.get():
            self._save_cfg()

    # ----- sector right-click menu ----------------------------------------
    def _sector_menu(self, event, name):
        s = self._find(name)
        if not s:
            return
        menu = tk.Menu(self.root, tearoff=0, bg=BG2, fg=FG,
                       activebackground="#3a3d44", activeforeground=FG)
        if s.get("type") == "pcaudio":
            menu.add_command(label="Change audio source…",
                             command=lambda n=name: self._change_audio_source(n))
            menu.add_separator()
        menu.add_command(label=f"Remove “{name}”",
                         command=lambda n=name: self._remove_one(n))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _change_audio_source(self, name):
        s = self._find(name)
        if not s or s.get("type") != "pcaudio":
            return
        dlg = ChangeDeviceDialog(self.root, s.get("output_device"))
        if dlg.result is None:
            return
        new_dev = dlg.result
        if new_dev == s.get("output_device"):
            return
        if self.engine:
            self.engine.change_device(s, new_dev)   # mutates output_device, restarts
        else:
            s["output_device"] = new_dev
        self._save_cfg()
        self._set_status(f"[{name}] now capturing: {new_dev}")

    def _remove_one(self, name):
        if messagebox.askyesno("Confirm removal", f"Remove “{name}”?"):
            self._do_remove([name])

    # ----- sector drag-to-reorder -----------------------------------------
    def _drag_start(self, name, color, event):
        self._drag_name = name
        self._drag_shadow = None
        self._drag_color = COLOR_HEX.get(color, FG)

    def _column_at(self, x_root):
        """Return the stream name of the sector column under x_root, or None."""
        for frame, col_name, _hdr in getattr(self, "_sector_headers", []):
            try:
                left = frame.winfo_rootx()
                right = left + frame.winfo_width()
            except Exception:
                continue
            if left <= x_root <= right:
                return col_name
        return None

    def _drag_motion(self, event):
        """Show a floating 'shadow' label following the cursor and highlight the
        column it's hovering over, so the drag is visualized."""
        name = getattr(self, "_drag_name", None)
        if not name:
            return
        # Create the shadow lazily on first motion.
        if not getattr(self, "_drag_shadow", None):
            sh = tk.Toplevel(self.root)
            sh.overrideredirect(True)            # no title bar / border
            sh.attributes("-topmost", True)
            try:
                sh.attributes("-alpha", 0.85)
            except Exception:
                pass
            tk.Label(sh, text=f"⠿ {name}", bg=BG2, fg=self._drag_color,
                     font=("Segoe UI", 10, "bold"), padx=10, pady=4,
                     relief="solid", borderwidth=1).pack()
            self._drag_shadow = sh
        # Move shadow near the cursor.
        self._drag_shadow.geometry(f"+{event.x_root + 12}+{event.y_root + 8}")
        # Highlight the hovered target column header.
        over = self._column_at(event.x_root)
        for _frame, col_name, hdr in getattr(self, "_sector_headers", []):
            if col_name == over and col_name != name:
                hdr.config(bg="#3a3d44")          # highlighted drop target
            else:
                hdr.config(bg=BG2)

    def _destroy_shadow(self):
        sh = getattr(self, "_drag_shadow", None)
        if sh is not None:
            try:
                sh.destroy()
            except Exception:
                pass
        self._drag_shadow = None
        # Reset any header highlights.
        for _frame, _col_name, hdr in getattr(self, "_sector_headers", []):
            try:
                hdr.config(bg=BG2)
            except Exception:
                pass

    def _drag_drop(self, event):
        """On header release, find which column the pointer is over and move the
        dragged stream to that position in self.streams (persisted + rebuilt)."""
        name = getattr(self, "_drag_name", None)
        self._drag_name = None
        self._destroy_shadow()
        if not name or not getattr(self, "_sector_headers", None):
            return
        target = self._column_at(event.x_root)
        if target is None or target == name:
            return
        self._reorder_streams(name, target)

    def _reorder_streams(self, name, target_name):
        """Move stream `name` to the target column's position. Direction-aware:
        dropping onto a column to the RIGHT lands after it; to the LEFT, before
        it. So the dragged column ends up exactly where you dropped it."""
        names = [s["name"] for s in self.streams]
        if name == target_name or name not in names or target_name not in names:
            return
        src = names.index(name)
        dragging_right = src < names.index(target_name)
        moving = self.streams.pop(src)
        # Recompute the target's index in the now-shortened list.
        t = next(i for i, s in enumerate(self.streams) if s["name"] == target_name)
        self.streams.insert(t + 1 if dragging_right else t, moving)
        self._save_cfg()
        self._rebuild_body()

    # ----- engine startup --------------------------------------------------
    def _start_engine(self):
        try:
            self.engine = core.Engine(
                self.cfg,
                on_line=lambda n, c, t, ts: self.events.put(("line", (n, c, t, ts))),
                on_status=lambda m: self.events.put(("status", m)),
                console=False, file_logging=True, enable_audio=True,
            )
            # TTS start/end callbacks -> marshal to UI thread to highlight the
            # line being read aloud.
            self.engine.tts_on_start = lambda text: self.events.put(("speak_start", text))
            self.engine.tts_on_end = lambda text: self.events.put(("speak_end", text))
            # Expand saved keyword presets into the engine's match list on startup
            # (config stores preset labels + free-text separately).
            self.engine.set_tts_keywords(self._effective_keywords())
            self.engine.load_model()
            self.engine.start_streams(self.streams)
            self.events.put(("ready", None))
        except Exception as e:
            self.events.put(("status", f"ENGINE ERROR: {e}"))

    # ----- event pump (UI thread) -----------------------------------------
    def _drain_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "line":
                    self._append_line(*payload)
                elif kind == "status":
                    self._set_status(payload)
                elif kind == "ready":
                    self._set_status("Model ready. Listening.")
                    self._refresh_listen_choices()
                    self._dismiss_splash()
                    self.root.after(400, self._maybe_prompt_login)
                elif kind == "model_done":
                    ok, msg, target = payload
                    self.model_combo.config(state="readonly")
                    if ok:
                        self._save_cfg()                 # persist the chosen model
                    else:
                        # Revert the dropdown to whatever is actually loaded.
                        self.model_var.set(self.engine.cfg.get("model", "large-v3"))
                    self._set_status(msg)
                elif kind == "update_result":
                    results, manual = payload
                    self._handle_update_result(results, manual)
                elif kind == "speak_start":
                    self._highlight_spoken(payload, True)
                elif kind == "speak_end":
                    self._highlight_spoken(payload, False)
                elif kind == "app_update_result":
                    info, manual = payload
                    self._handle_app_update_result(info, manual)
                elif kind == "app_dl_progress":
                    if getattr(self, "_update_dialog", None):
                        self._update_dialog.set_progress(*payload)
                elif kind == "app_dl_done":
                    if getattr(self, "_update_dialog", None):
                        self._update_dialog.on_done()
                    self._launch_installer_and_quit(payload)
                elif kind == "app_dl_error":
                    if getattr(self, "_update_dialog", None):
                        self._update_dialog.on_error(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _unit_color(self, unit):
        """Stable color for a unit/call sign: assigned once, reused all session."""
        if unit not in self._unit_colors:
            self._unit_colors[unit] = UNIT_PALETTE[len(self._unit_colors) % len(UNIT_PALETTE)]
        return self._unit_colors[unit]

    def _passes_filter(self, unit):
        """A line is shown if no unit filter is active or its unit matches."""
        return self.filter_unit is None or unit == self.filter_unit

    def _append_line(self, name, color, text, ts):
        # Compute the call sign ONCE here so coloring is consistent on replay.
        unit = core.extract_callsign(text, self.extra_prefixes)
        # Always record to history (filter affects display only, not the record).
        self.history.append((name, color, text, ts, unit))
        if self._passes_filter(unit):
            self._render_line(name, color, text, ts, unit, autoscroll=True)

    def _line_fg(self, stream_color, unit):
        """Resolve the foreground color for a line per the current color mode."""
        if self.color_mode.get() == "unit":
            return self._unit_color(unit) if unit else NO_UNIT_COLOR
        return COLOR_HEX.get(stream_color, FG)

    def _active_text_widgets(self):
        """All transcript Text widgets currently on screen (unified or sectors)."""
        if self.view_mode.get() == "unified":
            return [self.unified] if getattr(self, "unified", None) else []
        return list(self.sector_panels.values())

    # Highlight colors for TTS: green while being read, blue-grey once read
    # (the "already read to you" marker), which persists until the line scrolls
    # off the top of the transcript.
    _HL_SPEAKING = "#4a5a2a"   # muted green — line currently being spoken
    _HL_SPOKEN = "#2e3f5c"     # muted blue — line that finished being read

    def _highlight_spoken(self, text, on):
        """On speak_start (on=True): mark the line green ('speaking'). On
        speak_end (on=False): switch THAT line to blue ('spoken') and keep it —
        it stays until it scrolls off. Only the actively-spoken line is green;
        previously-read lines retain the blue marker."""
        for w in self._active_text_widgets():
            try:
                w.tag_config("speaking", background=self._HL_SPEAKING)
                w.tag_config("spoken", background=self._HL_SPOKEN)
                # 'speaking' must draw above 'spoken' where they'd overlap.
                w.tag_raise("speaking")
            except Exception:
                continue
            # Find the most recent line containing the spoken text.
            idx = w.search(text, "end", backwards=True, stopindex="1.0")
            if not idx:
                continue
            start, end = f"{idx} linestart", f"{idx} lineend+1c"
            if on:
                # Only scroll to the spoken line if the user is already tailing
                # the bottom; if they've scrolled up to read history, leave their
                # position alone (same rule as incoming lines).
                stick = self._at_bottom(w)
                w.tag_add("speaking", start, end)
                if stick:
                    w.see(idx)
            else:
                # Finished: clear the green on this line, apply the persistent blue.
                w.tag_remove("speaking", start, end)
                w.tag_add("spoken", start, end)

    def _bind_unit_click(self, widget, tag, unit):
        """Make a unit tag clickable -> filter to that unit (hand cursor + click)."""
        widget.tag_config(tag, underline=False)
        widget.tag_bind(tag, "<Enter>", lambda e, w=widget: w.config(cursor="hand2"))
        widget.tag_bind(tag, "<Leave>", lambda e, w=widget: w.config(cursor=""))
        widget.tag_bind(tag, "<Button-1>", lambda e, u=unit: self.set_unit_filter(u))

    def _feed_location(self, name):
        """The city/state to anchor map links for a feed (its 'location' field,
        else the global default_location in config, else None)."""
        s = self._find(name)
        if s and s.get("location"):
            return s["location"]
        return self.cfg.get("default_location") or None

    def _open_map(self, query, location):
        import webbrowser
        try:
            webbrowser.open(core.maps_url(query, location))
        except Exception:
            pass

    def _insert_message_text(self, widget, name, text, base_tags):
        """Insert the message body into `widget`, turning detected addresses into
        clickable Google-Maps links. `base_tags` are applied to normal text; each
        address also gets a unique clickable link tag."""
        addrs = core.extract_addresses(text)
        if not addrs:
            widget.insert("end", f"  {text}\n", base_tags)
            return
        widget.insert("end", "  ", base_tags)
        pos = 0
        location = self._feed_location(name)
        for span, query in addrs:
            i = text.find(span, pos)
            if i < 0:
                continue
            if i > pos:
                widget.insert("end", text[pos:i], base_tags)
            # Unique tag per link so each opens its own address.
            self._link_seq = getattr(self, "_link_seq", 0) + 1
            tag = f"addr:{self._link_seq}"
            widget.tag_config(tag, foreground=LINK_FG, underline=True)
            widget.tag_bind(tag, "<Enter>", lambda e, w=widget: w.config(cursor="hand2"))
            widget.tag_bind(tag, "<Leave>", lambda e, w=widget: w.config(cursor=""))
            widget.tag_bind(tag, "<Button-1>",
                            lambda e, q=query, loc=location: self._open_map(q, loc))
            widget.insert("end", span, base_tags + (tag,))
            pos = i + len(span)
        if pos < len(text):
            widget.insert("end", text[pos:], base_tags)
        widget.insert("end", "\n", base_tags)

    @staticmethod
    def _at_bottom(widget):
        """True if the text widget is scrolled to (or within a line of) the end.

        Used to decide whether an incoming line should auto-scroll: only pin to
        the bottom when the user is already there. If they've scrolled up to
        read history, we leave their position untouched. An empty widget reports
        yview() == (0.0, 1.0), so the very first line still scrolls."""
        try:
            return widget.yview()[1] >= 0.999
        except Exception:
            return True

    def _render_line(self, name, color, text, ts, unit=None, autoscroll=True):
        """
        Draw a single transcript line into the active view (no history write).

        Coloring: the message TEXT is white by default and takes the speaker's
        color when a unit/call sign is identified. The LABEL/prefix is colored
        per the 'Color by' mode (and is clickable in unit mode to filter).
        """
        label_fg = self._line_fg(color, unit)
        label_tag = f"u:{unit}" if self.color_mode.get() == "unit" else f"s:{name}"
        label = unit if (self.color_mode.get() == "unit" and unit) else name
        clickable = self.color_mode.get() == "unit" and unit
        # Message text: speaker color when known, else plain white (no tag).
        text_tag = ()
        if unit:
            text_tag = (f"tu:{unit}",)

        if self.view_mode.get() == "unified":
            t = self.unified
            stick = self._at_bottom(t)
            t.tag_config(label_tag, foreground=label_fg)
            t.tag_config("ts", foreground=MUTED)
            if unit:
                t.tag_config(text_tag[0], foreground=self._unit_color(unit))
            if clickable:
                self._bind_unit_click(t, label_tag, unit)
            t.configure(state="normal")
            t.insert("end", f"[{ts}] ", ("ts",))
            t.insert("end", f"{label:<16}", (label_tag,))
            self._insert_message_text(t, name, text, text_tag)
            if autoscroll and stick:
                t.see("end")
            t.configure(state="disabled")
        else:
            t = self.sector_panels.get(name)
            if t is not None:
                stick = self._at_bottom(t)
                t.tag_config(label_tag, foreground=label_fg)
                t.tag_config("ts2", foreground=MUTED)
                if unit:
                    t.tag_config(text_tag[0], foreground=self._unit_color(unit))
                if clickable:
                    self._bind_unit_click(t, label_tag, unit)
                t.configure(state="normal")
                # In unit mode, prefix the line with the clickable unit label.
                if clickable:
                    t.insert("end", f"[{ts}] ", ("ts2",))
                    t.insert("end", f"{unit}", (label_tag,))
                    self._insert_message_text(t, name, text, text_tag)
                else:
                    t.insert("end", f"[{ts}] ", ("ts2",))
                    self._insert_message_text(t, name, text, text_tag)
                if autoscroll and stick:
                    t.see("end")
                t.configure(state="disabled")

    def _replay_history(self):
        """Re-draw retained lines after the body is rebuilt. Disabled feeds'
        panels don't exist in sectors view, so those lines are simply skipped."""
        active = {s["name"] for s in self._enabled_streams()}
        for name, color, text, ts, unit in self.history:
            if self.view_mode.get() == "sectors" and name not in active:
                continue
            if not self._passes_filter(unit):
                continue
            self._render_line(name, color, text, ts, unit, autoscroll=False)
        # Snap each visible panel to the bottom once at the end.
        if self.view_mode.get() == "unified":
            self.unified.see("end")
        else:
            for t in self.sector_panels.values():
                t.see("end")

    def set_unit_filter(self, unit):
        """Show only lines from `unit` (None clears). Triggered by clicking a unit."""
        self.filter_unit = unit
        self._update_filter_bar()
        self._build_view()   # redraw with filter applied (replays from history)

    def clear_unit_filter(self):
        self.set_unit_filter(None)

    def _on_color_mode_change(self):
        # Leaving unit mode: a unit filter would be unclearable by clicking, so drop it.
        if self.color_mode.get() != "unit" and self.filter_unit:
            self.filter_unit = None
            self._update_filter_bar()
        self._rebuild_body()

    def _update_filter_bar(self):
        if self.filter_unit:
            self.filter_label.config(text=f"  Filter: {self.filter_unit}")
            self.filter_clear_btn.config(state="normal")
        else:
            self.filter_label.config(text="  Filter: (all units)")
            self.filter_clear_btn.config(state="disabled")

    def _change_font(self, delta):
        """Resize the shared transcript font (delta=+1/-1, or 0 to reset).
        Updates every panel live and persists the size to config.json."""
        new = FONT_DEFAULT if delta == 0 else self.font_size.get() + delta
        new = max(FONT_MIN, min(FONT_MAX, new))
        if new == self.font_size.get() and delta != 0:
            return
        self.font_size.set(new)
        self.transcript_font.configure(size=new)   # live update of all panels
        self.font_label.config(text=str(new))
        self.cfg["font_size"] = new
        self._save_cfg()

    def _set_status(self, msg):
        self.status.config(text=f"  {msg}")

    def _clear_text(self):
        self.history.clear()
        if self.view_mode.get() == "unified":
            self.unified.configure(state="normal")
            self.unified.delete("1.0", "end")
            self.unified.configure(state="disabled")
        else:
            for t in self.sector_panels.values():
                t.configure(state="normal")
                t.delete("1.0", "end")
                t.configure(state="disabled")

    # ----- stream library: core operations (reused by all dialogs) ---------
    def _find(self, name):
        for s in self.streams:
            if s["name"] == name:
                return s
        return None

    def _enabled_streams(self):
        return [s for s in self.streams if core.is_enabled(s)]

    def _do_add(self, stream):
        """Add a feed to the ACTIVE streams (and remember it in the library).
        If the feed already exists but is disabled (toggled Off), re-enable it
        rather than rejecting. Returns True on success."""
        existing = self._find(stream["name"])
        if existing:
            if not core.is_enabled(existing):
                self._do_set_enabled(stream["name"], True)   # re-activate it
                return True
            messagebox.showwarning("Duplicate", f"A stream named '{stream['name']}' exists.")
            return False
        self.streams.append(stream)
        self._lib_upsert(stream)                # remember it permanently
        if self.engine and core.is_enabled(stream):
            self.engine.add_stream(stream)
        self._save_cfg()
        self._rebuild_body()
        return True

    def _do_remove(self, names):
        """Remove one or more feeds from the ACTIVE streams. They STAY in the
        library so they can be re-added later (use _lib_delete to forget entirely)."""
        names = set(names)
        for name in names:
            if self.engine:
                self.engine.remove_stream(name)
        self.streams = [s for s in self.streams if s["name"] not in names]
        self._save_cfg()
        self._rebuild_body()

    def _lib_delete(self, names):
        """Permanently forget feeds from the library (and stop them if active)."""
        names = set(names)
        for name in names:
            if self.engine:
                self.engine.remove_stream(name)
        self.streams = [s for s in self.streams if s["name"] not in names]
        self.library = [e for e in self.library if e["name"] not in names]
        self._save_cfg()
        self._rebuild_body()

    def _do_edit(self, old_name, new_entry):
        """Edit a feed (possibly renaming). Updates the library and, if the feed
        is currently active, restarts it live with the new settings."""
        new_name = new_entry["name"]
        if new_name != old_name and self._find(new_name):
            messagebox.showwarning("Duplicate", f"A stream named '{new_name}' exists.")
            return False
        # Update library.
        self.library = [e for e in self.library if e["name"] != old_name]
        self._lib_upsert(new_entry)
        # Update active stream if present (preserve enabled/disabled state).
        active = self._find(old_name)
        if active:
            was_enabled = core.is_enabled(active)
            if self.engine:
                self.engine.remove_stream(old_name)
            merged = dict(new_entry)
            merged["disabled"] = not was_enabled
            idx = self.streams.index(active)
            self.streams[idx] = merged
            if self.engine and was_enabled:
                self.engine.add_stream(merged)
        self._save_cfg()
        self._rebuild_body()
        return True

    def _do_set_enabled(self, name, enabled):
        """Toggle a saved feed on/off: start/stop it live and persist the flag."""
        s = self._find(name)
        if not s:
            return
        s["disabled"] = not enabled
        if self.engine:
            if enabled:
                self.engine.add_stream(s)
            else:
                self.engine.remove_stream(name)
        self._save_cfg()
        self._rebuild_body()

    # ----- stream management entry points ---------------------------------
    def _open_library(self):
        CatalogDialog(self.root, self)

    def _open_tts(self):
        TTSDialog(self.root, self)

    def _effective_keywords(self):
        """Combine checked keyword-preset synonyms with the free-text extras."""
        terms = expand_keyword_presets(self.tts_cfg.get("keyword_presets", []))
        terms += self.tts_cfg.get("keywords", [])
        # De-dupe, keep order.
        seen, out = set(), []
        for t in terms:
            tl = t.lower()
            if tl and tl not in seen:
                seen.add(tl); out.append(t)
        return out

    def _save_tts_cfg(self):
        """Persist TTS settings to config and push them to the running engine."""
        self.cfg["tts"] = self.tts_cfg
        self._save_cfg()
        e = self.engine
        if e:
            e.set_tts_voice(self.tts_cfg.get("voice"))
            e.set_tts_feeds(self.tts_cfg.get("feeds", []))
            e.set_tts_keywords(self._effective_keywords())   # presets + extras
            e.set_tts_mode(self.tts_cfg.get("mode", "feeds"))
            e.set_tts_enabled(self.tts_cfg.get("enabled", False))

    def _refresh_listen_choices(self):
        names = ["(none)"] + [s["name"] for s in self._enabled_streams()]
        self.listen_combo.config(values=names)
        if self.listen_var.get() not in names:
            self.listen_var.set("(none)")

    def _on_listen_change(self, _evt=None):
        if not self.engine:
            return
        if not self.engine.audio_available():
            self._set_status("Audio output unavailable on this system.")
            return
        sel = self.listen_var.get()
        self.engine.listen_to(None if sel == "(none)" else sel)
        self._set_status(f"Listening to: {sel}" if sel != "(none)" else "Audio muted.")

    def _on_model_change(self, _evt=None):
        """Switch Whisper model live. Loads on a bg thread; feeds keep running."""
        target = self.model_var.get()
        if not self.engine:
            return
        if target == self.engine.cfg.get("model"):
            return
        # Disable the picker while loading so we don't queue overlapping swaps.
        self.model_combo.config(state="disabled")
        self._set_status(f"Loading model '{target}' (feeds keep running)...")

        def done(ok, msg):
            # Marshal result back to the UI thread via the event queue.
            self.events.put(("model_done", (ok, msg, target)))

        threading.Thread(
            target=lambda: self.engine.set_model(target, on_done=done),
            daemon=True).start()

    # ----- update checks ---------------------------------------------------
    def _run_update_check(self, manual):
        """Background worker: query PyPI, post result to the event queue.
        manual=True -> always show a popup; manual=False -> quiet status only."""
        try:
            results = core.check_for_updates()
        except Exception as e:
            results = [{"package": "?", "installed": None, "latest": None,
                        "update_available": False, "error": str(e)}]
        self.events.put(("update_result", (results, manual)))

    def _check_updates_manual(self):
        self.update_btn.config(state="disabled")
        self._set_status("Checking for updates...")
        threading.Thread(target=self._run_update_check, args=(True,),
                         daemon=True).start()

    def _check_updates_auto(self):
        # Quiet check shortly after launch; never blocks startup.
        threading.Thread(target=self._run_update_check, args=(False,),
                         daemon=True).start()

    def _handle_update_result(self, results, manual):
        self.update_btn.config(state="normal")
        avail = [r for r in results if r.get("update_available")]
        offline = all(r.get("latest") is None for r in results)

        if avail:
            names = ", ".join(f"{r['package']} {r['installed']}->{r['latest']}"
                              for r in avail)
            self._set_status(f"Update available: {names}")
        elif offline:
            self._set_status("Update check: couldn't reach PyPI (offline?).")
        else:
            self._set_status("Up to date.")

        if not manual:
            return  # auto check: status line only, no popup

        # Manual: full popup with versions + the exact (manual) update command.
        lines = []
        for r in results:
            inst = r.get("installed") or "not installed"
            latest = r.get("latest") or "unknown (offline?)"
            flag = "  *** UPDATE ***" if r.get("update_available") else ""
            lines.append(f"{r['package']}:\n    installed {inst}  |  latest {latest}{flag}")
        body = "\n\n".join(lines)
        if avail:
            cmd = ".venv\\Scripts\\python.exe -E -m pip install -U " + \
                  " ".join(r["package"] for r in avail)
            body += ("\n\nTo update (run in the project folder, then restart):\n"
                     f"    {cmd}\n\nNote: updates are manual on purpose. After "
                     "updating, new Whisper models may appear in the Model dropdown.")
            messagebox.showinfo("Updates available", body)
        elif offline:
            messagebox.showwarning(
                "Update check", body + "\n\nCould not reach PyPI. Check your "
                "internet connection and try again.")
        else:
            messagebox.showinfo("Up to date", body + "\n\nYou're on the latest "
                                "released versions.")

    # ----- shutdown --------------------------------------------------------
    def _on_close(self):
        if self.engine:
            self.engine.shutdown()
        self.root.destroy()


def _set_app_user_model_id():
    """Give the app its own Windows taskbar identity so it shows OUR icon (not
    the generic python icon) and groups on its own. No-op off Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ScannerTranscriber.App")
    except Exception:
        pass


def _show_splash(root):
    """Create a borderless splash window with the official logo, centered, on
    top. Returns the Toplevel (or None if no logo) so the caller can dismiss it
    when ready. A "Loading..." label sits under the logo."""
    if not os.path.exists(SPLASH_LOGO):
        return None
    try:
        splash = tk.Toplevel(root)
        splash.overrideredirect(True)          # no title bar / border
        splash.configure(bg=BG)
        try:
            splash.attributes("-topmost", True)
        except Exception:
            pass
        # High-quality scale to fit within 60% of the screen (LANCZOS via Pillow).
        sw, sht = splash.winfo_screenwidth(), splash.winfo_screenheight()
        img = load_scaled_image(SPLASH_LOGO, max_w=int(sw * 0.6), max_h=int(sht * 0.6))
        if img is None:
            splash.destroy()
            return None
        splash._img = img                       # keep a reference
        tk.Label(splash, image=img, bg=BG, borderwidth=0).pack()
        tk.Label(splash, text="Loading speech model…", bg=BG, fg=MUTED,
                 font=("Segoe UI", 10)).pack(pady=(0, 10))
        splash.update_idletasks()
        w, h = splash.winfo_reqwidth(), splash.winfo_reqheight()
        x, y = (sw - w) // 2, (sht - h) // 2
        splash.geometry(f"{w}x{h}+{x}+{y}")
        return splash
    except Exception:
        return None


def main():
    core.enable_windows_ansi()
    _set_app_user_model_id()
    root = tk.Tk()
    try:
        ttk.Style().theme_use("clam")
    except Exception:
        pass
    # Show the splash and keep the main window hidden until the model is ready
    # (with a 2s minimum so the logo is always visible for at least that long).
    root.withdraw()
    splash = _show_splash(root)
    TranscriberGUI(root, splash=splash)
    root.mainloop()


if __name__ == "__main__":
    main()
