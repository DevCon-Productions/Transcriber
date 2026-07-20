"""Headless-ish smoke test of the GUI widget logic with a stubbed engine.

Verifies: window builds, lines render in both unified and sectors views,
view switching works, add/remove updates the listen choices. Does NOT load the
Whisper model or touch the network (Engine is stubbed).
"""
import tkinter as tk
import transcriber as core
import gui


class FakeEngine:
    def __init__(self, *a, **k):
        self.added = []
        self.removed = []
        self._listen = None
        self.cfg = {"model": "tiny"}
        self.set_model_calls = []
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
        "model": "tiny", "vad": {}, "filters": {},
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

        # --- TTS dialog: check-all / clear presets ---------------------------
        if core.tts_available():
            tdlg = gui.TTSDialog(app.root, app)
            tdlg._set_all_presets(True)
            allc = tdlg._collect()
            results["tts_check_all"] = (
                len(allc["keyword_presets"]) == len(gui.KEYWORD_PRESETS))
            tdlg._set_all_presets(False)
            results["tts_clear_all"] = (tdlg._collect()["keyword_presets"] == [])
            tdlg.destroy()
        else:
            results["tts_check_all"] = True   # skip if no voices installed
            results["tts_clear_all"] = True

        # --- Help + About dialogs build without error ------------------------
        h = gui.HelpDialog(app.root)
        results["help_has_guide"] = ("USER GUIDE" in gui.HELP_TEXT)
        h.destroy()
        a = gui.AboutDialog(app.root)
        results["about_version"] = (gui.APP_VERSION == "1.3")
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
