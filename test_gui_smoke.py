"""Headless-ish smoke test of the GUI widget logic with a stubbed engine.

Verifies: window builds, lines render in both unified and sectors views,
view switching works, add/remove updates the listen choices. Does NOT load the
Whisper model or touch the network (Engine is stubbed).
"""
import time as _time_mod
import zipfile
import tkinter as tk
import transcriber as core
import gui


class FakeClips:
    """Stands in for Engine.clips. The GUI reads its dir/retention and, for the
    past-day picker, asks it which clips exist -- delegate that to a real
    ClipStore over the same directory so the lookup logic is genuinely exercised."""
    def __init__(self):
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="fakeclips_")
        self.enabled = False
        self.retention_days = 7

    def _real(self):
        return core.ClipStore({}, clip_dir=self.dir)

    def clip_map(self, day=None):
        return self._real().clip_map(day)

    def clip_info(self, clip_id):
        return self._real().clip_info(clip_id)

    def path_for(self, clip_id):
        # Bundling reads the clip files through this; without it the writer
        # would treat every clip as missing.
        return self._real().path_for(clip_id)

    def load_pcm(self, clip_id):
        return self._real().load_pcm(clip_id)


class FakeEngine:
    def __init__(self, *a, **k):
        self.added = []
        self.removed = []
        self._listen = None
        self.cfg = {"model": "tiny"}
        self.set_model_calls = []
        self.set_device_calls = []
        self.clips = FakeClips()
        self.recording = {}
        self.played = []
        self.play_sources = []      # None for live clips, the bundle when open
        self.spoken = []            # lines sent to read-aloud
        self.cancels = 0
        self.tts_ok = True          # flip off to model "no voice installed"
        self._clip_busy = False
        self._tts_busy = False
    def set_recording(self, name, on): self.recording[name] = on
    def set_clips_enabled(self, on): self.clips.enabled = on
    def play_clip(self, clip_id, source=None):
        self.played.append(clip_id)
        self.play_sources.append(source)
        self._clip_busy = True          # play-through polls until this clears
        return True

    # -- play-through: the transport drives these, so they model the real
    # engine's "start it, then poll until it's finished" contract.
    def clip_playing(self):
        return self._clip_busy

    def speak_now(self, text):
        if not self.tts_ok:
            return False
        self.spoken.append(text)
        self._tts_busy = True
        return True

    def tts_busy(self):
        return self._tts_busy

    def cancel_speech(self):
        self._tts_busy = False
        self.cancels += 1

    def finish_current(self):
        """Test hook: pretend whatever was sounding has just ended."""
        self._clip_busy = False
        self._tts_busy = False
    def stop_clip(self): pass
    def load_model(self): pass
    def start_streams(self, streams): self.added += [s["name"] for s in streams]
    def add_stream(self, s): self.added.append(s["name"]); return True
    def remove_stream(self, n): self.removed.append(n); return True
    def listen_to(self, n): self._listen = n; return True
    def audio_available(self): return True
    def now_listening(self): return self._listen
    def set_model(self, name, on_done=None):
        # Simulate success: record + update cfg + invoke callback synchronously.
        self.set_model_calls.append(name)
        self.cfg["model"] = name
        if on_done:
            on_done(True, f"Switched to '{name}'.")
    def set_device(self, device, on_done=None):
        # Simulate success: record + update cfg + invoke callback synchronously.
        self.set_device_calls.append(device)
        self.cfg["device"] = device
        if on_done:
            on_done(True, f"Switched to {device}.")
    def change_device(self, stream, new_device):
        if isinstance(new_device, str):
            stream["output_device"] = new_device
        else:
            stream["device"] = new_device
        self.changed_device = (stream["name"], new_device)
        return True
    def shutdown(self): pass


def run():
    # Stub the engine so no model/network is needed.
    core.Engine = FakeEngine

    # Give it a known stream set.
    core.load_config = lambda *a, **k: {
        "model": "tiny", "engine": "ct2", "vad": {}, "filters": {},
        "streams": [
            {"name": "West", "url": "https://audio.broadcastify.com/25008.mp3",
             "color": "cyan", "provider": "broadcastify"},
            {"name": "East", "url": "https://audio.broadcastify.com/12345.mp3",
             "color": "yellow", "provider": "broadcastify"},
        ],
    }
    # Don't actually overwrite config.json during the test.
    gui.TranscriberGUI._save_cfg = lambda self: None

    root = tk.Tk()
    app = gui.TranscriberGUI(root)
    results = {}

    def checks():
        # Stub ALL message dialogs up front so no real modal ever blocks the
        # after()-loop (a real showwarning mid-test can tear down root).
        _popups = []
        gui.messagebox.showinfo = lambda *a, **k: _popups.append(("info", a))
        gui.messagebox.showwarning = lambda *a, **k: _popups.append(("warn", a))
        gui.messagebox.showerror = lambda *a, **k: _popups.append(("err", a))
        gui.messagebox.askyesno = lambda *a, **k: True

        # Default view should be "sectors" when config has no view_mode set.
        results["view_default_sectors"] = (app.view_mode.get() == "sectors")

        # Force unified for the unified-view assertions below.
        app.view_mode.set("unified")
        # Simulate transcript lines arriving.
        app._append_line("West", "cyan", "engine 14, show me en route", "10:00:01")
        app._append_line("East", "yellow", "copy that, on scene", "10:00:02")
        content = app.unified.get("1.0", "end")
        results["unified_has_west"] = "West" in content and "en route" in content
        results["unified_has_east"] = "East" in content and "on scene" in content

        # Switch to sectors view and ensure per-panel routing.
        app.view_mode.set("sectors")
        app._append_line("West", "cyan", "sector-west-line", "10:00:03")
        app._append_line("East", "yellow", "sector-east-line", "10:00:04")
        west = app.sector_panels["West"].get("1.0", "end")
        east = app.sector_panels["East"].get("1.0", "end")
        results["sector_west_isolated"] = ("sector-west-line" in west
                                           and "sector-east-line" not in west)
        results["sector_east_isolated"] = ("sector-east-line" in east
                                           and "sector-west-line" not in east)

        # Listen choices reflect streams.
        app._refresh_listen_choices()
        vals = app.listen_combo.cget("values")
        results["listen_choices"] = ("West" in vals and "East" in vals
                                     and "(none)" in vals)

        # Listen selection routes to engine.
        app.listen_var.set("East")
        app._on_listen_change()
        results["listen_routed"] = (app.engine.now_listening() == "East")

        # --- Library operations (new feature) ---------------------------------
        # Add a feed via the core op (as the dialogs do).
        app._do_add({"name": "Fire", "url": "https://audio.broadcastify.com/23058.mp3",
                     "color": "red", "provider": "broadcastify"})
        results["add_in_library"] = (app._find("Fire") is not None)
        results["add_started_engine"] = ("Fire" in app.engine.added)

        # Disable a feed: should stop in engine, stay in library, drop from sectors.
        app._do_set_enabled("West", False)
        results["disable_persists"] = (app._find("West").get("disabled") is True)
        results["disable_stops_engine"] = ("West" in app.engine.removed)
        app.view_mode.set("sectors")
        results["disabled_not_rendered"] = ("West" not in app.sector_panels)
        # Disabled feed should not be a listen choice.
        app._refresh_listen_choices()
        results["disabled_not_listenable"] = ("West" not in app.listen_combo.cget("values"))

        # Re-enable: restarts in engine, reappears. (Reset the recorded adds
        # first so this is self-contained regardless of earlier checks.)
        app.engine.added = []
        app._do_set_enabled("West", True)
        results["reenable_starts_engine"] = ("West" in app.engine.added)
        results["reenable_rendered"] = ("West" in app.sector_panels)

        # Multi-remove: delete two feeds at once.
        app._do_remove(["East", "Fire"])
        results["multi_remove_gone"] = (app._find("East") is None
                                        and app._find("Fire") is None)
        results["multi_remove_engine"] = ("East" in app.engine.removed
                                          and "Fire" in app.engine.removed)
        results["remaining_is_west"] = ([s["name"] for s in app.streams] == ["West"])

        # --- Transcript retention across rebuilds (new fix) -------------------
        app._clear_text()                              # reset history + widgets
        app.view_mode.set("unified")
        app._append_line("West", "cyan", "RETAIN-ME-123", "11:00:00")
        results["history_recorded"] = any(
            "RETAIN-ME-123" in h[2] for h in app.history)
        # Switch view -> body rebuilds; the line must replay, not vanish.
        app.view_mode.set("sectors")
        west_panel = app.sector_panels.get("West")
        replayed_sectors = west_panel.get("1.0", "end") if west_panel else ""
        results["retained_after_view_switch"] = "RETAIN-ME-123" in replayed_sectors
        # Switch back to unified -> still there.
        app.view_mode.set("unified")
        results["retained_back_to_unified"] = "RETAIN-ME-123" in app.unified.get("1.0", "end")
        # Clear wipes history so it does NOT come back on next rebuild.
        app._clear_text()
        app.view_mode.set("sectors")
        wp = app.sector_panels.get("West")
        results["clear_wipes_history"] = (wp is None) or ("RETAIN-ME-123" not in wp.get("1.0", "end"))

        # --- Color-by-unit mode (new feature) --------------------------------
        app._clear_text()
        app.view_mode.set("unified")
        app.color_mode.set("stream")          # reset to a known state
        app._append_line("West", "cyan", "Adam 33 for code 2.", "12:00:00")
        app._append_line("West", "cyan", "King Tom George, 9-0-5-1.", "12:00:01")
        # history stores the extracted unit as the 5th element.
        units = [h[4] for h in app.history]
        results["unit_detected"] = ("ADAM 33" in units)
        results["plate_not_detected"] = (units[1] is None)  # the plate line

        # Switch to unit coloring: same unit -> stable color; two different units
        # -> different colors.
        app.color_mode.set("unit")
        c_adam = app._unit_color("ADAM 33")
        c_adam2 = app._unit_color("ADAM 33")
        c_engine = app._unit_color("ENGINE 14")
        results["unit_color_stable"] = (c_adam == c_adam2)
        results["unit_colors_differ"] = (c_adam != c_engine)
        # The unit tag must exist in the unified widget after rendering in unit mode.
        tags = app.unified.tag_names()
        results["unit_tag_rendered"] = ("u:ADAM 33" in tags)

        # --- Message TEXT colored by speaker (new) ---------------------------
        # A "tu:<unit>" tag colors the message body in the speaker's color, and
        # it must exist whenever a unit is identified -- in BOTH color modes.
        app._clear_text()
        app.color_mode.set("stream")
        app.view_mode.set("unified")
        app._append_line("West", "cyan", "Adam 33 for code 2.", "13:30:00")
        results["text_colored_in_stream_mode"] = ("tu:ADAM 33" in app.unified.tag_names())
        # The speaker text color should equal that unit's assigned color.
        results["text_color_matches_unit"] = (
            str(app.unified.tag_cget("tu:ADAM 33", "foreground")) == app._unit_color("ADAM 33"))
        # A line with no unit should NOT get a tu: tag (stays white/default).
        app._append_line("West", "cyan", "Okay, copy, thank you.", "13:30:01")
        results["no_unit_no_text_tag"] = not any(
            t.startswith("tu:") and t != "tu:ADAM 33" for t in app.unified.tag_names())

        # --- Removal consolidated into Manage Feeds only ---------------------
        results["no_standalone_remove"] = not hasattr(app, "_remove_stream")
        import gui as _gui
        results["select_dialog_gone"] = not hasattr(_gui, "SelectStreamsDialog")

        # --- Catalog: _do_add of catalog entries lands in library -------------
        from gui import FEED_CATALOG
        results["catalog_has_all_used"] = all(
            n in {e["name"] for e in FEED_CATALOG}
            for n in ["Cleveland West", "Cleveland Citywide",
                      "Cleveland Fire/EMS", "Westlake/WestCom"])
        entry = {k: v for k, v in FEED_CATALOG[-1].items() if k != "desc"}
        before = app._find(entry["name"]) is not None
        app._do_add(entry)
        results["catalog_add_works"] = (not before) and (app._find(entry["name"]) is not None)

        # --- Persistent library: remove from active keeps it in library ------
        libfeed = {"name": "LibTest", "url": "http://lib", "color": "cyan",
                   "provider": "broadcastify"}
        app._do_add(libfeed)
        results["lib_added_active"] = (app._find("LibTest") is not None)
        results["lib_added_library"] = (app._lib_find("LibTest") is not None)
        app._do_remove(["LibTest"])
        results["lib_remove_drops_active"] = (app._find("LibTest") is None)
        results["lib_remove_keeps_library"] = (app._lib_find("LibTest") is not None)  # the fix!
        # Re-add from library works.
        app._do_add(dict(app._lib_find("LibTest")))
        results["lib_readd"] = (app._find("LibTest") is not None)
        # Delete from library forgets it entirely.
        app._lib_delete(["LibTest"])
        results["lib_delete_forgets"] = (app._lib_find("LibTest") is None
                                         and app._find("LibTest") is None)

        # --- New feed -> LIBRARY ONLY, not auto-transcribed (bug fix) --------
        newfeed = {"name": "LibOnly", "url": "http://x", "color": "cyan",
                   "provider": "broadcastify"}
        ok = app._do_add_to_library(newfeed)
        results["libonly_added_to_library"] = (ok and app._lib_find("LibOnly") is not None)
        results["libonly_not_active"] = (app._find("LibOnly") is None)  # must NOT auto-start
        # Duplicate library add is rejected.
        results["libonly_dup_rejected"] = (app._do_add_to_library(newfeed) is False)

        # --- Toggled-Off feed reads as not-active; Add re-enables it ----------
        app._do_add({"name": "TogTest", "url": "http://t", "color": "red",
                     "provider": "broadcastify"})
        app._do_set_enabled("TogTest", False)               # turn it Off
        results["toggled_off_not_enabled"] = (
            core.is_enabled(app._find("TogTest")) is False)
        app._do_add(dict(app._lib_find("TogTest")))          # library "Add" again
        results["readd_reenables"] = (core.is_enabled(app._find("TogTest")) is True)
        app._lib_delete(["TogTest"]); app._lib_delete(["LibOnly"])  # cleanup

        # --- Library reorder (drag rows), direction-aware + persists ----------
        for n in ("RA", "RB", "RC", "RD"):
            app._do_add_to_library({"name": n, "url": f"http://{n}", "color": "cyan"})
        liborder = lambda: [e["name"] for e in app.library if e["name"] in
                            {"RA", "RB", "RC", "RD"}]
        # Drag RA DOWN onto RC -> RA lands in RC's slot: RB, RC, RA, RD.
        app._reorder_library("RA", "RC")
        results["lib_reorder_down"] = (liborder() == ["RB", "RC", "RA", "RD"])
        # Drag RD UP onto RB (drag up -> insert before target): RD, RB, RC, RA.
        app._reorder_library("RD", "RB")
        results["lib_reorder_up"] = (liborder() == ["RD", "RB", "RC", "RA"])
        # No-op: onto itself / unknown.
        before = liborder()
        app._reorder_library("RB", "RB")
        app._reorder_library("ZZ", "RB")
        results["lib_reorder_noop"] = (liborder() == before)
        # Persistence: the order lives in app.library; _save_cfg writes it to
        # cfg["feed_library"] verbatim. Verify that mapping without a disk write.
        app.cfg["feed_library"] = app.library      # what _save_cfg does
        cfg_order = [e["name"] for e in app.cfg["feed_library"]
                     if e["name"] in {"RA", "RB", "RC", "RD"}]
        results["lib_reorder_persisted"] = (cfg_order == liborder())
        for n in ("RA", "RB", "RC", "RD"):
            app._lib_delete([n])

        # --- Edit a feed: change URL + rename, library + active both update ---
        app._do_add({"name": "EditMe", "url": "http://old", "color": "red",
                     "provider": "broadcastify"})
        app._do_edit("EditMe", {"name": "EditMe", "url": "http://new",
                                "color": "red", "provider": "broadcastify"})
        results["edit_url"] = (app._find("EditMe")["url"] == "http://new"
                               and app._lib_find("EditMe")["url"] == "http://new")
        # Rename: old name gone, new present, in both active and library.
        app._do_edit("EditMe", {"name": "Renamed", "url": "http://new",
                                "color": "red", "provider": "broadcastify"})
        results["edit_rename_active"] = (app._find("EditMe") is None
                                         and app._find("Renamed") is not None)
        results["edit_rename_library"] = (app._lib_find("EditMe") is None
                                          and app._lib_find("Renamed") is not None)
        app._lib_delete(["Renamed"])   # cleanup

        # --- Unit click-filter -----------------------------------------------
        app._clear_text()
        app.color_mode.set("unit")
        app._append_line("West", "cyan", "Adam 33 for code 2.", "13:00:00")
        app._append_line("West", "cyan", "Engine 14 on scene.", "13:00:01")
        app._append_line("West", "cyan", "Okay, copy, thank you.", "13:00:02")  # no unit
        # Filter to ADAM 33: only that line should be visible.
        app.set_unit_filter("ADAM 33")
        shown = app.unified.get("1.0", "end")
        results["filter_keeps_match"] = ("code 2" in shown)
        results["filter_hides_others"] = ("Engine 14" not in shown
                                          and "thank you" not in shown)
        results["filter_history_intact"] = (len(app.history) == 3)  # record untouched
        # Clearing shows everything again.
        app.clear_unit_filter()
        shown2 = app.unified.get("1.0", "end")
        results["clear_filter_restores"] = ("code 2" in shown2 and "Engine 14" in shown2
                                            and "thank you" in shown2)
        # Switching to stream color mode auto-drops the filter.
        app.set_unit_filter("ADAM 33")
        app.color_mode.set("stream")
        results["stream_mode_drops_filter"] = (app.filter_unit is None)

        # --- Model picker (new) ----------------------------------------------
        from gui import MODEL_CHOICES
        results["model_choices_present"] = ("large-v3" in MODEL_CHOICES
                                            and "distil-large-v3" in MODEL_CHOICES)
        # Model list is engine-aware: the x64 big models on ct2, small English
        # models on whisper.cpp/ARM (the big ones are multi-GB + slow on CPU).
        results["model_choices_ct2"] = (
            _gui.model_choices({"engine": "ct2"}) == MODEL_CHOICES)
        arm_choices = _gui.model_choices({"engine": "whispercpp"})
        results["model_choices_arm_small"] = ("base.en" in arm_choices
                                              and "large-v3" not in arm_choices)
        # The configured model is ALWAYS selectable -- otherwise a config running
        # base.en on ARM could never be re-picked from the x64 list (the old trap).
        results["model_choices_keeps_current"] = (
            "base.en" in _gui.model_choices({"engine": "ct2"}, "base.en"))
        results["model_choices_no_dupe"] = (
            _gui.model_choices({"engine": "ct2"}, "large-v3").count("large-v3") == 1)
        # Changing the model triggers engine.set_model with the chosen name.
        app.model_var.set("large-v3")
        app._on_model_change()
        # Picker is disabled while loading.
        results["picker_disabled_loading"] = (str(app.model_combo.cget("state")) == "disabled")
        results["set_model_called"] = (app.engine.set_model_calls == ["large-v3"])
        # Drain the model_done event the fake engine queued -> picker re-enabled.
        app._drain_events()
        results["picker_reenabled"] = (str(app.model_combo.cget("state")) == "readonly")
        results["model_persisted"] = (app.engine.cfg.get("model") == "large-v3")
        # Selecting the already-current model is a no-op (no extra set_model call).
        app._on_model_change()
        results["no_redundant_swap"] = (app.engine.set_model_calls == ["large-v3"])

        # --- Device picker (new) ---------------------------------------------
        from gui import DEVICE_CHOICES
        results["device_choices_present"] = (DEVICE_CHOICES == ["Auto", "GPU", "CPU"])
        # Built on the ct2 backend (cfg engine='ct2'); hidden on whisper.cpp/ARM.
        results["device_combo_built"] = (app.device_combo is not None)
        app.device_var.set("CPU")
        app._on_device_change()
        results["device_persisted_to_cfg"] = (app.cfg.get("device") == "cpu")
        results["device_combo_disabled_loading"] = (
            str(app.device_combo.cget("state")) == "disabled")
        results["set_device_called"] = (app.engine.set_device_calls == ["cpu"])
        app._drain_events()   # process device_done -> re-enable the picker
        results["device_combo_reenabled"] = (
            str(app.device_combo.cget("state")) == "readonly")

        # --- Update check result handling (new) ------------------------------
        # Capture popups instead of showing them.
        popups = []
        gui.messagebox.showinfo = lambda *a, **k: popups.append(("info", a))
        gui.messagebox.showwarning = lambda *a, **k: popups.append(("warn", a))

        # Auto check (manual=False): status only, NO popup.
        app._handle_update_result(
            [{"package": "faster-whisper", "installed": "1.2.1", "latest": "9.9.9",
              "update_available": True},
             {"package": "ctranslate2", "installed": "4.7.2", "latest": "4.7.2",
              "update_available": False}], manual=False)
        results["auto_no_popup"] = (len(popups) == 0)
        results["auto_status_shows_update"] = ("Update available" in app.status.cget("text"))

        # Manual check with an update: popup shown, button re-enabled.
        app._handle_update_result(
            [{"package": "faster-whisper", "installed": "1.2.1", "latest": "9.9.9",
              "update_available": True}], manual=True)
        results["manual_popup_shown"] = (len(popups) == 1 and popups[0][0] == "info")
        results["btn_reenabled"] = (str(app.update_btn.cget("state")) == "normal")

        # Manual check, all up to date: still a popup, says up to date.
        popups.clear()
        app._handle_update_result(
            [{"package": "faster-whisper", "installed": "1.2.1", "latest": "1.2.1",
              "update_available": False}], manual=True)
        results["uptodate_popup"] = (len(popups) == 1)

        # Manual check, offline (latest None): warning popup.
        popups.clear()
        app._handle_update_result(
            [{"package": "faster-whisper", "installed": "1.2.1", "latest": None,
              "update_available": False}], manual=True)
        results["offline_warns"] = (len(popups) == 1 and popups[0][0] == "warn")

        # --- Font size control (new) -----------------------------------------
        from gui import FONT_MIN, FONT_MAX, FONT_DEFAULT
        start = app.font_size.get()
        app._change_font(+1)
        results["font_increases"] = (app.font_size.get() == start + 1)
        # The shared font object resizes -> all panels reflect it live.
        results["font_obj_updated"] = (app.transcript_font.cget("size") == start + 1)
        results["font_label_updated"] = (app.font_label.cget("text") == str(start + 1))
        app._change_font(-1)
        results["font_decreases"] = (app.font_size.get() == start)
        # Reset (delta 0) -> default.
        app._change_font(0)
        results["font_reset"] = (app.font_size.get() == FONT_DEFAULT)
        # Clamps at the maximum.
        for _ in range(100):
            app._change_font(+1)
        results["font_clamps_max"] = (app.font_size.get() == FONT_MAX)
        # Clamps at the minimum.
        for _ in range(100):
            app._change_font(-1)
        results["font_clamps_min"] = (app.font_size.get() == FONT_MIN)
        # Persisted to cfg.
        results["font_persisted"] = (app.cfg.get("font_size") == FONT_MIN)

        # --- PC-audio source type (new) --------------------------------------
        app.color_mode.set("stream")
        app.view_mode.set("sectors")
        pc = {"name": "PC Audio", "type": "pcaudio", "device": 0, "color": "white"}
        added_ok = app._do_add(pc)
        results["pcaudio_added"] = (added_ok and app._find("PC Audio") is not None)
        # It should count as enabled (it has a device) and render as a sector.
        results["pcaudio_enabled"] = (core.is_enabled(app._find("PC Audio")) is True)
        results["pcaudio_renders_sector"] = ("PC Audio" in app.sector_panels)
        # Engine got an add call for it.
        results["pcaudio_started"] = ("PC Audio" in app.engine.added)
        # A pcaudio entry with no device must NOT be enabled.
        results["pcaudio_no_device_disabled"] = (
            core.is_enabled({"name": "X", "type": "pcaudio", "color": "white"}) is False)

        # --- per-application source (type=app) -------------------------------
        appsrc = {"name": "MyApp", "type": "app", "pid": 4321,
                  "app_name": "vlc.exe", "color": "magenta"}
        added_app = app._do_add(appsrc)
        results["app_added"] = (added_app and app._find("MyApp") is not None)
        results["app_enabled"] = (core.is_enabled(app._find("MyApp")) is True)
        results["app_renders_sector"] = ("MyApp" in app.sector_panels)
        results["app_started"] = ("MyApp" in app.engine.added)
        results["app_no_pid_disabled"] = (
            core.is_enabled({"name": "Y", "type": "app", "color": "white"}) is False)

        # --- Sector column reorder, DIRECTION-AWARE (new) --------------------
        def fresh():
            app.streams = [
                {"name": "A", "url": "http://a", "color": "cyan"},
                {"name": "B", "url": "http://b", "color": "green"},
                {"name": "C", "url": "http://c", "color": "red"},
                {"name": "D", "url": "http://d", "color": "blue"},
            ]
        order = lambda: [s["name"] for s in app.streams]

        # REGRESSION: leftmost (A) dragged RIGHT onto B must actually move.
        # A lands in B's slot -> B, A, C, D.
        fresh(); app._reorder_streams("A", "B")
        results["drag_right_leftmost"] = (order() == ["B", "A", "C", "D"])

        # Drag A all the way right onto D -> B, C, D, A.
        fresh(); app._reorder_streams("A", "D")
        results["drag_right_far"] = (order() == ["B", "C", "D", "A"])

        # Drag rightmost (D) LEFT onto B -> A, D, B, C.
        fresh(); app._reorder_streams("D", "B")
        results["drag_left"] = (order() == ["A", "D", "B", "C"])

        # Drag onto an adjacent neighbor to the left: C onto B -> A, C, B, D.
        fresh(); app._reorder_streams("C", "B")
        results["drag_left_adjacent"] = (order() == ["A", "C", "B", "D"])

        # No-op cases: unknown name, or moving onto itself.
        fresh(); before = order()
        app._reorder_streams("ZZ", "A")
        app._reorder_streams("A", "A")
        results["reorder_noop_safe"] = (order() == before)

        # --- Right-click change-audio-source (output-device by name) ---------
        # Stub the device dialog to return a chosen OUTPUT DEVICE NAME.
        class FakeDevDlg:
            def __init__(self, parent, current): self.result = "Speakers (Realtek(R) Audio)"
        gui.ChangeDeviceDialog = FakeDevDlg
        app.streams = [{"name": "TV", "type": "pcaudio",
                        "output_device": "Old Speakers", "color": "white"}]
        app._change_audio_source("TV")
        results["change_source_updates_device"] = (
            app._find("TV")["output_device"] == "Speakers (Realtek(R) Audio)")
        results["change_source_calls_engine"] = (
            getattr(app.engine, "changed_device", None) == ("TV", "Speakers (Realtek(R) Audio)"))
        # Changing a URL stream's source is a no-op (guard).
        app.streams = [{"name": "Feed", "url": "http://x", "color": "cyan"}]
        app.engine.changed_device = None
        app._change_audio_source("Feed")
        results["change_source_url_noop"] = (app.engine.changed_device is None)

        # --- TTS: highlight the spoken line ----------------------------------
        app._clear_text()
        app.view_mode.set("unified")
        app._append_line("West", "cyan", "Adam 33 en route", "14:00:00")
        app._append_line("West", "cyan", "structure fire on Fifth", "14:00:01")
        app._highlight_spoken("structure fire on Fifth", True)
        rng = app.unified.tag_ranges("speaking")
        hl = app.unified.get(rng[0], rng[1]) if rng else ""
        results["tts_highlight_line"] = ("structure fire on Fifth" in hl
                                         and "Adam 33" not in hl)
        app._highlight_spoken("structure fire on Fifth", False)
        # After speaking: 'speaking' (green) cleared, 'spoken' (blue) persists.
        results["tts_highlight_clear"] = (len(app.unified.tag_ranges("speaking")) == 0)
        spk = app.unified.tag_ranges("spoken")
        results["tts_spoken_persists"] = (len(spk) > 0 and
            "structure fire on Fifth" in app.unified.get(spk[0], spk[1]))
        results["tts_two_colors"] = (
            gui.TranscriberGUI._HL_SPEAKING != gui.TranscriberGUI._HL_SPOKEN)

        # --- Clickable address -> map links ----------------------------------
        app._clear_text()
        app.view_mode.set("unified")
        app.streams = [{"name": "West", "url": "http://x", "color": "cyan",
                        "location": "Cleveland, OH"}]
        app._append_line("West", "cyan", "units to 3658 East 149th Street", "15:00:00")
        link_tags = [t for t in app.unified.tag_names() if t.startswith("addr:")]
        results["addr_link_created"] = (len(link_tags) == 1)
        if link_tags:
            rng = app.unified.tag_ranges(link_tags[0])
            results["addr_link_text"] = (
                "3658 East 149th Street" in app.unified.get(rng[0], rng[1]))
            results["addr_link_underlined"] = (
                str(app.unified.tag_cget(link_tags[0], "underline")) in ("1", "True"))
        else:
            results["addr_link_text"] = False
            results["addr_link_underlined"] = False
        # Line with no address makes no new link.
        app._append_line("West", "cyan", "copy that, clear", "15:00:01")
        results["addr_no_false_link"] = (
            len([t for t in app.unified.tag_names() if t.startswith("addr:")]) == 1)
        # Feed location resolves for map query.
        results["addr_feed_location"] = (app._feed_location("West") == "Cleveland, OH")

        # --- TTS keyword presets: expand + combine with extras ----------------
        from gui import expand_keyword_presets, KEYWORD_PRESETS
        results["presets_exist"] = (len(KEYWORD_PRESETS) >= 6)
        shooting = expand_keyword_presets(["Shooting"])
        results["preset_expands"] = ("gunfire" in shooting and "shots fired" in shooting)
        results["preset_unknown_safe"] = (expand_keyword_presets(["Nope"]) == [])
        # _effective_keywords merges preset synonyms + free-text, de-duped.
        app.tts_cfg = {"keyword_presets": ["Shooting", "Fire"],
                       "keywords": ["euclid ave", "gunfire"]}  # 'gunfire' dup on purpose
        eff = app._effective_keywords()
        results["effective_has_preset"] = ("gunfire" in eff and "flames" in eff)
        results["effective_has_extra"] = ("euclid ave" in eff)
        results["effective_dedupes"] = (eff.count("gunfire") == 1)

        # --- TTS dialog: check-all / clear presets + engine picker -----------
        if core.tts_available():
            tdlg = gui.TTSDialog(app.root, app)
            tdlg._set_all_presets(True)
            allc = tdlg._collect()
            results["tts_check_all"] = (
                len(allc["keyword_presets"]) == len(gui.KEYWORD_PRESETS))
            tdlg._set_all_presets(False)
            results["tts_clear_all"] = (tdlg._collect()["keyword_presets"] == [])
            # Engine picker: Auto is always offered, only usable engines beyond it,
            # and _collect() records the chosen engine id.
            labels = [lab for _id, lab in tdlg._engine_choices]
            results["tts_engine_has_auto"] = ("Auto (recommended)" in labels)
            results["tts_engine_only_usable"] = (
                [i for i, _l in tdlg._engine_choices if i != "auto"]
                == core.available_tts_engines())
            results["tts_engine_collected"] = (allc.get("engine") in
                                               ["auto"] + core.available_tts_engines())
            # Switching engine refreshes the voice list to that engine's voices.
            if core.available_tts_engines():
                first = gui._TTS_ID_TO_LABEL[core.available_tts_engines()[0]]
                tdlg.engine_var.set(first)
                vlist = list(tdlg.voice_combo.cget("values"))
                want = [v[0] for v in core.list_tts_voices(
                    engine=tdlg._selected_engine())]
                results["tts_engine_voice_refresh"] = (vlist == want)
            else:
                results["tts_engine_voice_refresh"] = True
            tdlg.destroy()
        else:
            for k in ("tts_check_all", "tts_clear_all", "tts_engine_has_auto",
                      "tts_engine_only_usable", "tts_engine_collected",
                      "tts_engine_voice_refresh"):
                results[k] = True   # skip if no voices installed

        # --- Help + About dialogs build without error ------------------------
        h = gui.HelpDialog(app.root)
        results["help_has_guide"] = ("USER GUIDE" in gui.HELP_TEXT)
        h.destroy()
        a = gui.AboutDialog(app.root)
        results["about_version"] = (gui.APP_VERSION == "1.4-arm64")
        a.destroy()

        # --- Broadcastify login dialog ---------------------------------------
        captured = {}
        d = gui.BroadcastifyLoginDialog(app.root, username="preset_user",
                                        password="preset_pass",
                                        on_save=lambda u, p: captured.update(u=u, p=p))
        results["login_prefills_user"] = (d.user_var.get() == "preset_user")
        results["login_prefills_pw"] = (d.pw_var.get() == "preset_pass")
        results["login_pw_masked"] = (str(d.pw_entry.cget("show")) not in ("", "None"))
        # empty username/password -> save rejected (dialog stays, no callback)
        d.user_var.set("")
        d._save()
        results["login_rejects_empty"] = ("u" not in captured and d.winfo_exists())
        # valid values -> callback fires with entered creds
        d.user_var.set("newuser")
        d.pw_var.set("newpass")
        d._save()
        results["login_saves_values"] = (captured.get("u") == "newuser"
                                         and captured.get("p") == "newpass")

        # --- App update dialog + result handling -----------------------------
        info = {"available": True, "current": "1.2", "latest": "1.5",
                "notes": "Shiny new things.", "html_url": "https://x/rel",
                "asset_name": "Transcriber-Setup-1.5.exe",
                "asset_url": "https://x/dl.exe", "asset_size": 500 * (1 << 20)}
        dl = {"clicked": False}
        ud = gui.AppUpdateDialog(app.root, info,
                                 on_download=lambda: dl.update(clicked=True),
                                 on_view=lambda: None)
        results["update_shows_version"] = ("1.5" in ud._status.cget("text") + " 1.5")
        ud._start()
        results["update_download_fires"] = (dl["clicked"] is True)
        results["update_btn_disabled"] = (str(ud._dl_btn.cget("state")) == "disabled")
        ud.set_progress(250 * (1 << 20), 500 * (1 << 20))   # 50%
        results["update_progress_pct"] = (int(ud._pbar.cget("value")) == 50)
        ud.on_error("boom")                                 # re-enables buttons
        results["update_error_reenables"] = (str(ud._dl_btn.cget("state")) == "normal")
        ud.destroy()

        # result handler: up-to-date path (manual messagebox is stubbed)
        app._update_dialog = None
        app._handle_app_update_result(
            {"available": False, "current": "1.2", "latest": "1.2"}, manual=False)
        results["update_uptodate_no_dialog"] = (app._update_dialog is None)
        # available path -> opens a dialog
        app._handle_app_update_result(info, manual=False)
        results["update_available_opens_dialog"] = (app._update_dialog is not None)
        if app._update_dialog:
            app._update_dialog.destroy()

        # --- Feed list export / import ---------------------------------------
        import json
        import os
        import tempfile
        tmp = tempfile.mkdtemp(prefix="guifeeds_")
        exported = os.path.join(tmp, "feeds.json")
        gui.filedialog.asksaveasfilename = lambda *a, **k: exported
        app._export_feeds()
        doc = json.load(open(exported, encoding="utf-8"))
        names = [f["name"] for f in doc["feeds"]]
        results["export_wrote_file"] = (doc["format"] == core.FEED_EXPORT_FORMAT
                                        and names == [e["name"] for e in app.library])
        results["export_no_engine_keys"] = ("model" not in doc and "device" not in doc)

        # Import a list with one new feed and one that collides by name.
        incoming = os.path.join(tmp, "incoming.json")
        core.export_feeds(incoming, [
            {"name": "Imported Feed", "url": "https://audio.example/9.mp3",
             "color": "magenta", "provider": "broadcastify"},
            {"name": "West", "url": "https://audio.example/CHANGED.mp3",
             "color": "blue", "provider": "broadcastify"},
        ])
        gui.filedialog.askopenfilename = lambda *a, **k: incoming

        class FakeConflict:
            mode = "skip"
            def __init__(self, *a, **k):
                self.result = FakeConflict.mode
        RealConflictDialog = gui.ImportConflictDialog     # exercised for real below
        gui.ImportConflictDialog = FakeConflict

        before_west = dict(app._lib_find("West") or {})
        app._import_feeds()                                   # mode = skip
        results["import_adds_new"] = (app._lib_find("Imported Feed") is not None)
        results["import_skip_keeps_mine"] = (
            app._lib_find("West").get("url") == before_west.get("url"))
        # Imported feeds are saved only -- they must not start transcribing.
        results["import_does_not_start"] = ("Imported Feed" not in app.engine.added)

        FakeConflict.mode = "rename"
        app._import_feeds()
        results["import_rename_suffixes"] = (app._lib_find("West (2)") is not None)

        FakeConflict.mode = "replace"
        app._import_feeds()
        results["import_replace_overwrites"] = (
            app._lib_find("West")["url"].endswith("CHANGED.mp3"))

        # A file that isn't a feed list surfaces an error, not a traceback.
        junk = os.path.join(tmp, "junk.json")
        json.dump({"nope": 1}, open(junk, "w"))
        gui.filedialog.askopenfilename = lambda *a, **k: junk
        errs_before = len([p for p in _popups if p[0] == "err"])
        app._import_feeds()
        results["import_bad_file_errors"] = (
            len([p for p in _popups if p[0] == "err"]) == errs_before + 1)

        # Cancelling the file picker is a no-op.
        gui.filedialog.askopenfilename = lambda *a, **k: ""
        lib_before = len(app.library)
        app._import_feeds()
        results["import_cancel_noop"] = (len(app.library) == lib_before)

        # --- Transcript -> PDF ------------------------------------------------
        pdf_path = os.path.join(tmp, "out.pdf")
        gui.filedialog.asksaveasfilename = lambda *a, **k: pdf_path

        class FakePdfDlg:
            spec = ("East", "session", [])
            def __init__(self, *a, **k):
                self.result = FakePdfDlg.spec
        RealPdfDialog = gui.PdfExportDialog               # exercised for real below
        gui.PdfExportDialog = FakePdfDlg

        app._append_line("East", "yellow", "pdf-session-line", "11:00:00")
        app._export_pdf()
        blob = open(pdf_path, "rb").read()
        results["pdf_session_written"] = blob.startswith(b"%PDF-1.4")
        results["pdf_session_content"] = (b"pdf-session-line" in blob
                                          and b"11:00:00" in blob)
        results["pdf_titled_by_feed"] = (b"(East) Tj" in blob)

        # Log-backed export reads the day files for that feed.
        logdir = os.path.join(tmp, "logs")
        os.makedirs(logdir)
        with open(os.path.join(logdir, "East-20260719.log"), "w",
                  encoding="utf-8") as f:
            f.write("[09:00:00] pdf-log-line\n")
        real_lookup = core.log_files_for
        core.log_files_for = lambda name, log_dir=logdir: real_lookup(name, logdir)
        FakePdfDlg.spec = ("East", "logs", ["20260719"])
        pdf2 = os.path.join(tmp, "out2.pdf")
        gui.filedialog.asksaveasfilename = lambda *a, **k: pdf2
        app._export_pdf()
        blob2 = open(pdf2, "rb").read()
        results["pdf_logs_written"] = (b"pdf-log-line" in blob2)
        # The header reports the real first/last transmission, dated from the
        # log's filename -- not just the day, and not a bare line count.
        results["pdf_header_span"] = (b"2026-07-19 09:00:00" in blob2)
        results["pdf_header_counts"] = (b"1 transmission)" in blob2)
        results["pdf_header_bold_title"] = (b"/F2 17.0 Tf" in blob2
                                            and b"(East) Tj" in blob2)
        core.log_files_for = real_lookup

        # A feed with nothing to export informs the user instead of writing
        # ("Imported Feed" was never started, so it has no transcript lines).
        # showinfo was re-stubbed further up, so capture it fresh here.
        FakePdfDlg.spec = ("Imported Feed", "session", [])
        infos = []
        gui.messagebox.showinfo = lambda *a, **k: infos.append(a)
        pdf3 = os.path.join(tmp, "never.pdf")
        gui.filedialog.asksaveasfilename = lambda *a, **k: pdf3
        app._export_pdf()
        results["pdf_empty_informs"] = (len(infos) == 1
                                        and not os.path.exists(pdf3))

        # --- The real (normally modal) dialogs build and apply correctly ------
        # simpledialog.Dialog blocks in wait_window(); bypass just that so the
        # widgets are really constructed and the apply() logic is really run.
        class LivePdfDlg(RealPdfDialog):
            def wait_window(self, *a, **k): pass

        class LiveConflictDlg(RealConflictDialog):
            def wait_window(self, *a, **k): pass

        core.log_files_for = lambda name, log_dir=logdir: real_lookup(name, logdir)
        pd = LivePdfDlg(app.root, app, ["East", "West"], preselect="East")
        pd.grab_release()
        results["pdfdlg_preselects"] = (pd.feed.get() == "East")
        # East has one log day (created above) -> listed, and logs stay selectable.
        results["pdfdlg_lists_days"] = (list(pd.daylist.get(0, "end")) == ["2026-07-19"])
        results["pdfdlg_days_enabled"] = (str(pd.daylist.cget("state")) == "normal")
        # No day selected == every day.
        pd.apply()
        results["pdfdlg_all_days"] = (pd.result == ("East", "logs", ["20260719"]))
        # A feed with no logs falls back to the session and greys the day list.
        pd.feed.set("West")
        results["pdfdlg_no_logs_fallback"] = (pd.source.get() == "session"
                                              and str(pd.daylist.cget("state")) == "disabled")
        pd.apply()
        results["pdfdlg_session_result"] = (pd.result == ("West", "session", []))
        pd.destroy()
        core.log_files_for = real_lookup

        # The Feeds window renders a row per library feed, each with a PDF button,
        # plus the Export/Import buttons -- build it once so a bad widget shows up.
        cat = gui.CatalogDialog(app.root, app)
        row_buttons = [w.cget("text") for row, _n in cat._rows
                       for w in row.winfo_children() if isinstance(w, tk.Button)]
        results["catalog_rows_built"] = (len(cat._rows) == len(app.library))
        results["catalog_row_has_pdf"] = (row_buttons.count("PDF") == len(cat._rows))
        footer = [w.cget("text") for f in cat.winfo_children()
                  for w in f.winfo_children() if isinstance(w, tk.Button)]
        results["catalog_has_list_buttons"] = ("Export list..." in footer
                                               and "Import list..." in footer)
        cat.destroy()

        cd = LiveConflictDlg(app.root, 5, ["West", "East"])
        cd.grab_release()
        results["conflictdlg_defaults_skip"] = (cd.mode.get() == "skip")
        cd.mode.set("rename")
        cd.apply()
        results["conflictdlg_applies"] = (cd.result == "rename")
        cd.destroy()

        # --- Clip recording ---------------------------------------------------
        # A line carrying a clip id renders a clickable speaker marker; one
        # without stays exactly as before.
        app.view_mode.set("unified")
        app._append_line("East", "yellow", "clip-marked line", "12:00:00",
                         "clip-abc")
        app._append_line("East", "yellow", "plain line", "12:00:01")
        content = app.unified.get("1.0", "end")
        results["clip_marker_drawn"] = ("🔊" in content and "clip-marked line" in content)
        results["clip_marker_tag"] = ("clip:clip-abc" in app.unified.tag_names())
        # Exactly one marker: the plain line must not get one.
        results["clip_marker_once"] = (content.count("🔊") == 1)
        # It survives a view rebuild (history carries the clip id, so the replay
        # re-draws the marker rather than losing it).
        app.view_mode.set("sectors")
        app.view_mode.set("unified")
        app._rebuild_body()
        results["clip_marker_replayed"] = ("🔊" in app.unified.get("1.0", "end"))

        # Clicking it asks the engine to play that clip.
        app._play_clip("clip-abc")
        results["clip_click_plays"] = (app.engine.played == ["clip-abc"])

        # History keeps the clip id alongside the line, and the PDF export still
        # unpacks the wider row.
        results["history_keeps_clip"] = any(
            row[-1] == "clip-abc" for row in app.history)

        # Per-feed opt-in round-trips through the Add/Edit dialog.
        class RecDlg(gui.AddStreamDialog):
            def wait_window(self, *a, **k): pass
        ad = RecDlg(app.root, title="Edit", initial={
            "name": "East", "url": "https://audio.example/e.mp3", "record": True})
        ad.grab_release()
        results["record_box_prefilled"] = (ad.record.get() is True)
        ad.apply()
        results["record_in_result"] = (ad.result.get("record") is True)
        ad.record.set(False)
        ad.apply()
        # Off means the key is simply absent, so config stays clean.
        results["record_off_omits_key"] = ("record" not in ad.result)
        ad.destroy()
        results["recording_feeds_helper"] = (app._recording_feeds() == set())

        # --- Restoring today's transcript from the logs -----------------------
        # Write a log for an active feed plus clips for two of its lines, then
        # re-run the restore and check the lines come back with their audio.
        rdir = tempfile.mkdtemp(prefix="restore_")
        rlogs = os.path.join(rdir, "logs")
        os.makedirs(rlogs)
        today = core.dt.datetime.now().strftime("%Y%m%d")
        active_feed = app._enabled_streams()[0]["name"]
        with open(os.path.join(rlogs, f"{core.safe_filename(active_feed)}-{today}.log"),
                  "w", encoding="utf-8") as f:
            f.write("[08:00:00] restored line one\n")
            f.write("[08:00:00] restored line two\n")   # same second, own clip
            f.write("[08:00:05] restored line three\n")
            f.write("garbage that is not a log line\n")

        rclips = os.path.join(rdir, "clips")
        os.makedirs(rclips)
        with open(os.path.join(rclips, f"index-{today}.jsonl"), "w",
                  encoding="utf-8") as f:
            for i, (ts, cid) in enumerate([("08:00:00", "cid-a"),
                                           ("08:00:00", "cid-b")]):
                f.write(json.dumps({"id": cid, "feed": active_feed, "day": today,
                                    "ts": ts, "text": f"x{i}"}) + "\n")

        real_logs_for = core.log_files_for
        core.log_files_for = lambda name, log_dir=rlogs: real_logs_for(name, rlogs)
        app.engine.clips.dir = rclips
        app.history.clear()
        app._restore_scrollback()
        core.log_files_for = real_logs_for

        restored = [r for r in app.history if r[0] == active_feed]
        results["restore_reads_log"] = (len(restored) == 3)
        results["restore_skips_garbage"] = all("garbage" not in r[2] for r in restored)
        results["restore_in_order"] = ([r[3] for r in restored] ==
                                       ["08:00:00", "08:00:00", "08:00:05"])
        # Same-second lines get DIFFERENT clips, in index order -- not the same one.
        results["restore_distinct_clips"] = ([r[5] for r in restored[:2]] ==
                                             ["cid-a", "cid-b"])
        # A line with no clip recorded stays unmarked.
        results["restore_no_clip_stays_none"] = (restored[2][5] is None)
        results["restore_counts"] = (app._restored == (3, 2))
        # The marker is drawn for restored lines too.
        app.view_mode.set("unified")
        app._rebuild_body()
        results["restore_marker_drawn"] = (
            app.unified.get("1.0", "end").count("🔊") == 2)

        # restore_lines = 0 disables it entirely.
        core.log_files_for = lambda name, log_dir=rlogs: real_logs_for(name, rlogs)
        app.cfg["restore_lines"] = 0
        app.history.clear()
        app._restore_scrollback()
        results["restore_disabled"] = (len(app.history) == 0)
        # A cap keeps the NEWEST lines.
        app.cfg["restore_lines"] = 1
        app._restore_scrollback()
        kept = [r for r in app.history if r[0] == active_feed]
        results["restore_caps_newest"] = (len(kept) == 1
                                          and kept[0][2] == "restored line three")
        core.log_files_for = real_logs_for
        app.cfg.pop("restore_lines", None)
        app.history.clear()

        # --- Selecting clips and exporting them as one MP3 --------------------
        app.past_day = None
        app.view_mode.set("unified")
        app.history.clear()
        app._rebuild_body()
        for i, (txt, cid) in enumerate([("alpha line", "cid-1"),
                                        ("bravo line", None),
                                        ("charlie line", "cid-2"),
                                        ("delta line", "cid-3")]):
            app._append_line(active_feed, "cyan", txt, f"12:00:0{i}", cid)

        u = app.unified
        u.tag_remove("sel", "1.0", "end")
        results["mp3_no_selection"] = (app._selected_clip_ids() == [])

        # --- Whole-row selection ---------------------------------------------
        # Clicking anywhere in a line must take the WHOLE line, and dragging
        # must extend by whole lines -- partial rows can't map to whole clips.
        app.root.update()
        mid = u.bbox("2.6")             # mid-text on line 2, not on the marker
        if mid:
            u.event_generate("<Button-1>", x=mid[0] + 2, y=mid[1] + 2)
            app.root.update()
            sel = u.tag_ranges("sel")
            results["row_click_whole_row"] = (
                bool(sel) and str(sel[0]) == "2.0"
                and u.get(sel[0], sel[1]) == u.get("2.0", "2.0 lineend"))
            # Drag down to line 4 -> lines 2..4 entirely.
            low = u.bbox("4.6")
            if low:
                u.event_generate("<B1-Motion>", x=low[0] + 2, y=low[1] + 2)
                app.root.update()
                sel2 = u.tag_ranges("sel")
                results["row_drag_extends"] = (
                    str(sel2[0]) == "2.0" and str(sel2[1]).startswith("4."))
                # Dragging back UP past the anchor selects upward, not nothing.
                top = u.bbox("1.6")
                if top:
                    u.event_generate("<B1-Motion>", x=top[0] + 2, y=top[1] + 2)
                    app.root.update()
                    sel3 = u.tag_ranges("sel")
                    results["row_drag_upward"] = (str(sel3[0]) == "1.0"
                                                  and str(sel3[1]).startswith("2."))
            u.event_generate("<ButtonRelease-1>", x=mid[0] + 2, y=mid[1] + 2)
        else:
            for k in ("row_click_whole_row", "row_drag_extends", "row_drag_upward"):
                results[k] = True       # no geometry in this environment

        # Clicking the 🔊 marker still plays rather than just selecting: the
        # widget-level Button-1 preempts tag bindings, so the click is dispatched
        # by _click_targets and this would silently regress without a check.
        app.engine.played.clear()
        marker = u.bbox("1.11")         # the marker sits after "[hh:mm:ss] "
        if marker:
            u.event_generate("<Button-1>", x=marker[0] + 1, y=marker[1] + 1)
            app.root.update()
            u.event_generate("<ButtonRelease-1>", x=marker[0] + 1, y=marker[1] + 1)
        results["row_click_marker_plays"] = (app.engine.played == ["cid-1"]
                                             if marker else True)
        # Map links keep working through the same dispatch.
        opened = []
        app._open_map = lambda q, loc: opened.append((q, loc))
        app._link_targets["addr:test"] = ("map", "100 Main St", "Cleveland, OH")
        u.tag_add("addr:test", "3.0", "3.5")
        hit = u.bbox("3.1")
        if hit:
            u.event_generate("<Button-1>", x=hit[0] + 1, y=hit[1] + 1)
            app.root.update()
        results["row_click_link_opens"] = (opened == [("100 Main St",
                                                       "Cleveland, OH")]
                                           if hit else True)
        u.tag_delete("addr:test")
        # An aircraft link goes to FlightRadar24, not to the map.
        urls = []
        app._open_url = lambda u_: urls.append(u_)
        app._link_targets["addr:air"] = ("air", "DL510", None)
        u.tag_add("addr:air", "4.0", "4.5")
        hit2 = u.bbox("4.1")
        if hit2:
            u.event_generate("<Button-1>", x=hit2[0] + 1, y=hit2[1] + 1)
            app.root.update()
        results["row_click_aircraft_opens"] = (
            urls == ["https://www.flightradar24.com/data/flights/dl510"]
            if hit2 else True)
        u.tag_delete("addr:air")
        u.tag_remove("sel", "1.0", "end")

        # Select lines 1-3: picks up cid-1 and cid-2, skips the clipless line,
        # and stops before cid-3.
        u.tag_add("sel", "1.0", "3.end")
        results["mp3_selection_maps"] = (app._selected_clip_ids() == ["cid-1", "cid-2"])
        # Order follows the transcript, not tag creation order.
        u.tag_remove("sel", "1.0", "end")
        u.tag_add("sel", "1.0", "end")
        results["mp3_selection_ordered"] = (app._selected_clip_ids() ==
                                            ["cid-1", "cid-2", "cid-3"])
        # A selection covering only a clipless line yields nothing.
        u.tag_remove("sel", "1.0", "end")
        u.tag_add("sel", "2.0", "2.end")
        results["mp3_selection_clipless"] = (app._selected_clip_ids() == [])

        # A real mouse drag must produce a selection -- the panes are disabled
        # Text widgets, and the whole feature rests on that still selecting.
        u.tag_remove("sel", "1.0", "end")
        app.root.update()
        u.event_generate("<Button-1>", x=5, y=5)
        u.event_generate("<B1-Motion>", x=200, y=40)
        u.event_generate("<ButtonRelease-1>", x=200, y=40)
        app.root.update()
        results["mp3_mouse_drag_selects"] = bool(u.tag_ranges("sel"))

        # Nothing selected -> tells the user how, and opens no save dialog.
        u.tag_remove("sel", "1.0", "end")
        infos3 = []
        gui.messagebox.showinfo = lambda *a, **k: infos3.append(a)
        asked = []
        gui.filedialog.asksaveasfilename = lambda *a, **k: asked.append(1) or ""
        app._export_selected_mp3()
        results["mp3_empty_selection_informs"] = (len(infos3) == 1 and not asked)

        # With a selection, cancelling the save dialog does nothing further.
        u.tag_add("sel", "1.0", "end")
        app._export_selected_mp3()
        results["mp3_cancel_noop"] = (len(asked) == 1)

        # The result handler reports success and flags purged clips.
        app._handle_mp3_done(True, "out.mp3", {"clips": 3, "seconds": 8.3,
                                               "missing": []})
        results["mp3_status_ok"] = ("3 clip" in app.status.cget("text")
                                    and "out.mp3" in app.status.cget("text"))
        app._handle_mp3_done(True, "out.mp3", {"clips": 2, "seconds": 5.0,
                                               "missing": ["x"]})
        results["mp3_status_missing"] = ("no longer on disk" in app.status.cget("text"))
        errs2 = []
        gui.messagebox.showerror = lambda *a, **k: errs2.append(a)
        app._handle_mp3_done(False, "boom", None)
        results["mp3_status_error"] = (len(errs2) == 1
                                       and "failed" in app.status.cget("text"))

        # The right-click menu enables/disables itself against the selection.
        u.tag_remove("sel", "1.0", "end")
        popped = {}
        class FakeMenu:
            def __init__(self, *a, **k): self.items = []
            def add_command(self, **k): self.items.append(k)
            def add_separator(self): pass
            def tk_popup(self, *a): popped["shown"] = self
            def grab_release(self): pass
        real_menu = gui.tk.Menu
        gui.tk.Menu = FakeMenu
        class Ev: x_root = y_root = 10
        app._transcript_menu(Ev(), u)
        results["mp3_menu_disabled_empty"] = (
            popped["shown"].items[0]["state"] == "disabled")
        u.tag_add("sel", "1.0", "end")
        app._transcript_menu(Ev(), u)
        results["mp3_menu_enabled_counts"] = (
            popped["shown"].items[0]["state"] == "normal"
            and "3 clips" in popped["shown"].items[0]["label"])
        gui.tk.Menu = real_menu
        u.tag_remove("sel", "1.0", "end")

        # SECTORS is the default view, and its columns use _sector_menu rather
        # than _transcript_menu -- so the export has to be on that menu too, or
        # right-clicking the transcript offers nothing where most users are.
        # Clear first: switching views replays history, so the unified block's
        # lines would otherwise be redrawn into this pane and counted too.
        app.history.clear()
        app.view_mode.set("sectors")
        app._rebuild_body()
        panel = app.sector_panels.get(active_feed)
        results["mp3_sector_panel_exists"] = (panel is not None)
        if panel is not None:
            for i, cid in enumerate(["s-1", "s-2"]):
                app._append_line(active_feed, "cyan", f"sector line {i}",
                                 f"13:00:0{i}", cid)
            panel.tag_add("sel", "1.0", "end")
            gui.tk.Menu = FakeMenu
            app._sector_menu(Ev(), active_feed)
            labels_menu = [it["label"] for it in popped["shown"].items]
            results["mp3_in_sector_menu"] = any("MP3" in l for l in labels_menu)
            results["mp3_sector_menu_counts"] = any("2 clips" in l
                                                    for l in labels_menu)
            results["mp3_sector_keeps_remove"] = any("Remove" in l
                                                     for l in labels_menu)
            # And the selection in a sector pane resolves, not just in unified.
            results["mp3_sector_selection_maps"] = (
                app._selected_clip_ids() == ["s-1", "s-2"])
            gui.tk.Menu = real_menu
            panel.tag_remove("sel", "1.0", "end")
        app.view_mode.set("unified")
        app.history.clear()
        app._rebuild_body()

        # --- Transcript bundles (.tscript) ------------------------------------
        app.past_day = None
        app.view_mode.set("unified")
        app.history.clear()
        app._rebuild_body()
        # Give the fake clip store real audio so a bundle can carry it.
        bstore = core.ClipStore({"clips": {"enabled": True}},
                                clip_dir=app.engine.clips.dir)
        bstore.start()
        import numpy as _np
        tone = (0.3 * _np.sin(_np.arange(core.SAMPLE_RATE // 2) * 0.05)
                ).astype(_np.float32)
        bid = bstore.save(active_feed, tone, text="bundle-line-one")
        bstore._q.join()
        bstore.stop()

        for i, (txt, cid) in enumerate([("bundle-line-one", bid),
                                        ("bundle-line-two", None)]):
            app._append_line(active_feed, "cyan", txt, f"14:00:0{i}", cid)

        # A selection recovers full rows -- text and timestamps, not just ids.
        u = app.unified
        u.tag_add("sel", "1.0", "end")
        rows_sel = app._selected_rows()
        results["bundle_selected_rows"] = (
            [r[1] for r in rows_sel] == ["bundle-line-one", "bundle-line-two"]
            and rows_sel[0][0] == "14:00:00" and rows_sel[0][3] == active_feed)
        results["bundle_rows_keep_clipless"] = (rows_sel[1][2] is None)

        bpath = os.path.join(tmp, "sel" + core.TRANSCRIPT_EXT)
        gui.filedialog.asksaveasfilename = lambda *a, **k: bpath
        app._export_selected_bundle()
        # The write runs on a worker thread; wait for the file to land.
        for _ in range(100):
            if os.path.exists(bpath) and os.path.getsize(bpath) > 0:
                break
            app.root.update()
            _time_mod.sleep(0.05)
        results["bundle_export_writes"] = (os.path.exists(bpath)
                                           and zipfile.is_zipfile(bpath))

        # Opening it switches into review mode, named after the file.
        gui.filedialog.askopenfilename = lambda *a, **k: bpath
        app._open_bundle()
        results["bundle_opened"] = (app.bundle is not None
                                    and app.past_day is not None)
        results["bundle_banner_names_file"] = (
            "sel" + core.TRANSCRIPT_EXT in app._bundle_name)
        shown_b = app.unified.get("1.0", "end")
        results["bundle_shows_lines"] = ("bundle-line-one" in shown_b
                                         and "bundle-line-two" in shown_b)
        results["bundle_shows_marker"] = (shown_b.count("🔊") == 1)
        # Clip playback must route to the BUNDLE, not the live store -- the
        # originals may not exist wherever this file gets opened.
        app.engine.played.clear()
        app.engine.play_sources.clear()
        app._play_clip(bid)
        results["bundle_plays_from_bundle"] = (
            app.engine.played == [bid]
            and app.engine.play_sources[0] is app.bundle)
        # MP3 export of an open bundle uses it as the source too.
        results["bundle_is_clip_store"] = (app._clip_store() is app.bundle)

        # Leaving review closes the archive (releases the file handle).
        opened_zip = app.bundle
        app._exit_past_day()
        results["bundle_closed_on_exit"] = (app.bundle is None
                                            and app.past_day is None)
        try:
            opened_zip._zip.read("manifest.json")
            results["bundle_handle_released"] = False
        except Exception:
            results["bundle_handle_released"] = True

        # A file that isn't a transcript reports it rather than blanking.
        junk_b = os.path.join(tmp, "junk" + core.TRANSCRIPT_EXT)
        with open(junk_b, "w", encoding="utf-8") as f:
            f.write("not a bundle")
        gui.filedialog.askopenfilename = lambda *a, **k: junk_b
        errs3 = []
        gui.messagebox.showerror = lambda *a, **k: errs3.append(a)
        app._open_bundle()
        results["bundle_bad_file_errors"] = (len(errs3) == 1
                                             and app.bundle is None)

        # Nothing selected -> explains itself, opens no save dialog.
        # Leaving review rebuilt the body, so the old widget is gone.
        u = app.unified
        u.tag_remove("sel", "1.0", "end")
        infos4 = []
        asked2 = []
        gui.messagebox.showinfo = lambda *a, **k: infos4.append(a)
        gui.filedialog.asksaveasfilename = lambda *a, **k: asked2.append(1) or ""
        app._export_selected_bundle()
        results["bundle_empty_selection_informs"] = (len(infos4) == 1
                                                     and not asked2)
        app.history.clear()
        app._rebuild_body()

        # --- Playing a reviewed transcript through -----------------------------
        # Drive the transport directly: start a line, declare it finished, and
        # pump Tk's after() queue so the poll loop advances exactly as it would
        # when a real clip or utterance ends.
        def pump(n=10):
            # Sleep THEN update, so a callback scheduled during the last update
            # still gets one to run in -- advancing chains after(10) onto
            # after(300), and updating first would leave the tail unprocessed.
            for _ in range(n):
                _time_mod.sleep(0.06)
                app.root.update()

        app.past_day = ("PlayFeed", "20260728")
        app._past_rows = [("09:00:00", "first line", "p1"),
                          ("09:00:01", "second line has no audio", None),
                          ("09:00:02", "third line", "p3")]
        app.bundle = None
        app._pb_reset()
        app._build_view()
        results["pb_bar_built"] = (app._pb_widgets is not None)
        results["pb_position_initial"] = (app._pb_widgets["pos"].cget("text")
                                          == "1 / 3")

        # Audio mode: plays clip 1, skips the clipless line, then plays clip 3.
        app.engine.played.clear()
        app._pb_start("audio")
        results["pb_audio_starts"] = (app.engine.played == ["p1"]
                                      and app._pb_running)
        results["pb_highlights_line"] = bool(
            app.unified.tag_ranges("pbline"))
        app.engine.finish_current()
        pump()
        # Line 2 has no clip, so it must not stall there -- it should already
        # have moved on to line 3.
        results["pb_skips_clipless"] = (app.engine.played == ["p1", "p3"])
        results["pb_position_advances"] = (app._pb_widgets["pos"].cget("text")
                                           == "3 / 3")
        app.engine.finish_current()
        pump()
        results["pb_stops_at_end"] = (not app._pb_running)
        results["pb_clears_highlight"] = (not app.unified.tag_ranges("pbline"))

        # Read-aloud mode speaks every line, including the one with no audio.
        app.engine.spoken.clear()
        app._pb_reset()
        app._pb_start("tts")
        results["pb_tts_starts"] = (app.engine.spoken == ["first line"])
        app.engine.finish_current()
        pump()
        results["pb_tts_reads_clipless"] = (
            app.engine.spoken == ["first line", "second line has no audio"])

        # Pause holds position and silences; resuming continues from there.
        app._pb_pause()
        results["pb_pause_stops"] = (not app._pb_running
                                     and app.engine.cancels > 0)
        held = app._pb_index
        app.engine.spoken.clear()
        app._pb_start("tts")
        results["pb_resumes_where_paused"] = (app._pb_index == held
                                              and app.engine.spoken == [
                                                  app._past_rows[held][1]])

        # Stop rewinds to the top and clears the highlight.
        app._pb_stop()
        results["pb_stop_rewinds"] = (app._pb_index == 0
                                      and not app._pb_running
                                      and not app.unified.tag_ranges("pbline"))

        # Selecting a row makes play start there instead of the beginning.
        app.unified.tag_add("sel", "3.0", "3.0 lineend")
        app.engine.played.clear()
        app._pb_start("audio")
        results["pb_starts_at_selection"] = (app._pb_index == 2
                                             and app.engine.played == ["p3"])
        app._pb_stop()
        app.unified.tag_remove("sel", "1.0", "end")

        # Switching mode mid-play silences the old one rather than doubling up.
        app.engine.played.clear()
        app.engine.spoken.clear()
        app._pb_start("audio")
        before_cancels = app.engine.cancels
        app._pb_start("tts")
        results["pb_mode_switch_silences"] = (
            app.engine.cancels > before_cancels and app._pb_mode == "tts")
        app._pb_stop()

        # With no voice installed, read-aloud says so instead of hanging.
        app.engine.tts_ok = False
        app._pb_start("tts")
        results["pb_tts_unavailable"] = (not app._pb_running
                                         and "text-to-speech" in
                                         app.status.cget("text"))
        app.engine.tts_ok = True

        # Leaving review must stop playback -- otherwise it keeps stepping
        # through rows that are no longer on screen, over the live feed.
        app._pb_start("audio")
        app._exit_past_day()
        results["pb_exit_stops"] = (not app._pb_running
                                    and app._pb_widgets is None)
        app.engine.finish_current()
        pump(3)
        results["pb_exit_no_further_play"] = (len(app.engine.played) <= 2)
        app._pb_reset()

        # --- Reviewing a past day ---------------------------------------------
        # Reuse the restore fixture's logs, plus an older text-only day.
        with open(os.path.join(rlogs, f"{core.safe_filename(active_feed)}-20260715.log"),
                  "w", encoding="utf-8") as f:
            f.write("[07:00:00] older day line\n")
        core.log_files_for = lambda name, log_dir=rlogs: real_logs_for(name, rlogs)

        class LivePastDlg(gui.PastDayDialog):
            def wait_window(self, *a, **k): pass
        pdlg = LivePastDlg(app.root, app, [active_feed])
        pdlg.grab_release()
        shown = list(pdlg.daylist.get(0, "end"))
        results["pastdlg_newest_first"] = (shown[0].startswith(today[:4] + "-")
                                           and "2026-07-15" in shown[-1])
        # The day with clips advertises them; the purged day doesn't.
        results["pastdlg_marks_audio"] = ("🔊" in shown[0] and "🔊" not in shown[-1])
        results["pastdlg_preselects"] = (pdlg.daylist.curselection() == (0,))
        pdlg.apply()
        results["pastdlg_result"] = (pdlg.result == (active_feed, today))
        pdlg.destroy()

        # Entering the past view: banner + lines + markers, live traffic held back.
        class FakePastDlg:
            spec = None
            def __init__(self, *a, **k):
                self.result = FakePastDlg.spec
        FakePastDlg.spec = (active_feed, today)
        gui.PastDayDialog = FakePastDlg
        app._open_past_day()
        results["past_mode_entered"] = (app.past_day == (active_feed, today))
        body_text = "".join(w.get("1.0", "end") for w in [app.unified])
        results["past_shows_day"] = ("restored line one" in body_text
                                     and "🔊" in body_text)
        # A line arriving while reviewing is recorded but not drawn.
        before = len(app.history)
        app._append_line(active_feed, "cyan", "live-during-past", "23:59:59")
        results["past_holds_live_line"] = (
            len(app.history) == before + 1
            and "live-during-past" not in app.unified.get("1.0", "end"))
        # A rebuild (font change, view toggle) must not knock us out of the day.
        app._rebuild_body()
        results["past_survives_rebuild"] = (
            app.past_day == (active_feed, today)
            and "restored line one" in app.unified.get("1.0", "end"))

        # Back to live: history replays, including what arrived while away.
        app._exit_past_day()
        results["past_mode_exited"] = (app.past_day is None)
        live_text = app.unified.get("1.0", "end") if app.view_mode.get() == "unified" \
            else "".join(t.get("1.0", "end") for t in app.sector_panels.values())
        results["past_exit_replays_live"] = ("live-during-past" in live_text)

        # A day with no transcript reports it instead of blanking the view.
        FakePastDlg.spec = (active_feed, "19700101")
        infos2 = []
        gui.messagebox.showinfo = lambda *a, **k: infos2.append(a)
        app._open_past_day()
        results["past_missing_day_informs"] = (len(infos2) == 1
                                               and app.past_day is None)
        gui.PastDayDialog = LivePastDlg.__bases__[0]
        core.log_files_for = real_logs_for

        # Recording dialog now owns only the on/off switch -- retention moved to
        # its own dialog, so it must not still be writing a retention value.
        class LiveClipSettings(gui.ClipSettingsDialog):
            def wait_window(self, *a, **k): pass
        app.cfg["clips"] = {"enabled": False, "retention_days": 7, "max_gb": 0}
        cs = LiveClipSettings(app.root, app)
        cs.grab_release()
        results["clipdlg_reads_cfg"] = (cs.enabled.get() is False)
        results["clipdlg_shows_policy"] = ("7 day" in cs.days_label.cget("text"))
        cs.enabled.set(True)
        cs.apply()
        results["clipdlg_applies"] = (cs.result == {"enabled": True})
        results["clipdlg_no_retention"] = ("retention_days" not in cs.result)
        cs.destroy()

        # --- Retention dialog -------------------------------------------------
        class LiveRetention(gui.RetentionDialog):
            def wait_window(self, *a, **k): pass
        app.cfg["log_retention_days"] = 14
        app.cfg["clips"] = {"enabled": True, "retention_days": 7, "max_gb": 0}
        rd = LiveRetention(app.root, app)
        rd.grab_release()
        results["retdlg_reads_both"] = (rd.log_days.get() == "14"
                                        and rd.clip_days.get() == "7")
        # No cap configured -> the box is off and its entry disabled.
        results["retdlg_cap_off"] = (rd.cap_on.get() is False
                                     and str(rd.cap_entry.cget("state"))
                                     == "disabled")
        rd.cap_on.set(True)
        rd._sync_state()
        results["retdlg_cap_enables"] = (str(rd.cap_entry.cget("state"))
                                         == "normal")
        rd.max_gb.set("0")
        results["retdlg_rejects_zero_cap"] = (rd.validate() is False)
        rd.log_days.set("-1")
        rd.max_gb.set("2")
        results["retdlg_rejects_negative"] = (rd.validate() is False)
        rd.log_days.set("30")
        rd.clip_days.set("3")
        rd.apply()
        results["retdlg_applies"] = (rd.result == {"log_days": 30.0,
                                                   "clip_days": 3.0,
                                                   "max_gb": 2.0})
        # Unticking the cap zeroes it rather than keeping a stale number.
        rd.cap_on.set(False)
        rd.apply()
        results["retdlg_cap_off_zeroes"] = (rd.result["max_gb"] == 0.0)
        rd.destroy()

        # The app writes both policies back and pushes them to the live store.
        class FakeRetention:
            spec = {"log_days": 21.0, "clip_days": 5.0, "max_gb": 1.5}
            def __init__(self, *a, **k):
                self.result = FakeRetention.spec
        real_ret = gui.RetentionDialog
        gui.RetentionDialog = FakeRetention
        app._open_retention()
        results["retention_saved_logs"] = (app.cfg["log_retention_days"] == 21.0)
        results["retention_saved_clips"] = (
            app.cfg["clips"]["retention_days"] == 5.0
            and app.cfg["clips"]["max_gb"] == 1.5)
        results["retention_pushed_to_engine"] = (
            app.engine.clips.retention_days == 5.0
            and app.engine.clips.max_gb == 1.5)
        results["retention_status"] = ("1.5 GB" in app.status.cget("text"))
        # Enabling recording must not clobber the retention just set.
        gui.RetentionDialog = real_ret

        # --- Feed groups: several feeds sharing one column --------------------
        app.past_day = None
        app.bundle = None
        app.history.clear()
        saved_streams = list(app.streams)
        app.streams = [
            {"name": "CLE Tower", "url": "u1", "color": "cyan",
             "group": "CLE ATC", "service": "atc"},
            {"name": "CLE Ground", "url": "u2", "color": "yellow",
             "group": "CLE ATC", "service": "atc"},
            {"name": "Cleveland West", "url": "u3", "color": "green"},
        ]
        app.view_mode.set("sectors")
        app._rebuild_body()
        results["grp_columns_merged"] = (app._active_groups() ==
                                         ["CLE ATC", "Cleveland West"])
        # Both ATC feeds must draw into the SAME widget, and the ungrouped feed
        # into its own.
        results["grp_shared_widget"] = (
            app.sector_panels["CLE Tower"] is app.sector_panels["CLE Ground"]
            and app.sector_panels["Cleveland West"]
            is not app.sector_panels["CLE Tower"])
        results["grp_marks_members"] = (app._grouped_feeds ==
                                        {"CLE Tower", "CLE Ground"})
        results["grp_known_groups"] = ("CLE ATC" in app._known_groups())

        # Lines from both channels land in the shared column, each labelled.
        app._append_line("CLE Tower", "cyan", "cleared to land", "15:00:00")
        app._append_line("CLE Ground", "yellow", "taxi via alpha", "15:00:01")
        app._append_line("Cleveland West", "green", "adam 33 responding", "15:00:02")
        shared = app.sector_panels["CLE Tower"].get("1.0", "end")
        own = app.sector_panels["Cleveland West"].get("1.0", "end")
        results["grp_both_channels_shown"] = ("cleared to land" in shared
                                              and "taxi via alpha" in shared)
        results["grp_channel_tagged"] = ("CLE Tower" in shared
                                         and "CLE Ground" in shared)
        results["grp_ungrouped_isolated"] = ("adam 33 responding" in own
                                             and "cleared to land" not in own)
        # An ungrouped column shouldn't waste width on a channel tag.
        results["grp_no_tag_when_alone"] = ("Cleveland West" not in own)

        # A selection in the shared column recovers rows from BOTH feeds.
        w = app.sector_panels["CLE Tower"]
        w.tag_add("sel", "1.0", "end")
        feeds_in_sel = {r[3] for r in app._selected_rows()}
        results["grp_selection_spans_feeds"] = (feeds_in_sel ==
                                                {"CLE Tower", "CLE Ground"})
        w.tag_remove("sel", "1.0", "end")

        # Dragging a grouped column moves its members together, not one of them.
        app._reorder_streams("CLE ATC", "Cleveland West")
        order = [s["name"] for s in app.streams]
        results["grp_reorder_moves_all"] = (order ==
                                            ["Cleveland West", "CLE Tower",
                                             "CLE Ground"])
        app.streams = saved_streams
        app.history.clear()
        app.view_mode.set("unified")
        app._rebuild_body()

        # --- Streams menu exposes the new commands ---------------------------
        menubar = app.root.nametowidget(app.root.cget("menu"))
        streams = app.root.nametowidget(menubar.entrycget(1, "menu"))
        labels = [streams.entrycget(i, "label")
                  for i in range(streams.index("end") + 1)
                  if streams.type(i) != "separator"]
        results["menu_has_export"] = ("Export feed list..." in labels)
        results["menu_has_import"] = ("Import feed list..." in labels)
        results["menu_has_pdf"] = ("Save transcript as PDF..." in labels)
        results["menu_has_recording"] = ("Audio recording..." in labels)
        results["menu_has_past_day"] = ("Open a past day..." in labels)

        app._on_close()

    # Tkinter swallows exceptions raised inside after() callbacks (prints to
    # stderr but doesn't propagate), which would silently truncate the checks.
    # Capture any exception so the test fails loudly instead of masking it.
    _err = {}

    def _run_checks():
        try:
            checks()
        except Exception:
            import traceback
            _err["tb"] = traceback.format_exc()
            try:
                app._on_close()
            except Exception:
                pass

    # checks() calls root.destroy via app._on_close when done; this is only a
    # backstop if it hangs. Keep it well above the real runtime so it never
    # truncates the checks (that was masking failures).
    root.after(300, _run_checks)
    root.after(30000, root.destroy)
    root.mainloop()

    if _err:
        print("CHECKS RAISED:\n" + _err["tb"])

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert not _err, f"GUI smoke test raised an exception:\n{_err.get('tb','')}"
    assert results and ok, "GUI smoke test failed"
    print("GUI SMOKE TEST: PASS")


if __name__ == "__main__":
    run()
