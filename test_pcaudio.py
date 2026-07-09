"""Unit tests for PC-audio capture support (resampler, device helpers,
is_enabled for pcaudio, and DeviceWorker wiring). No real audio device needed."""
import numpy as np
import transcriber as core


def run():
    r = {}

    # --- resampler ----------------------------------------------------------
    x48 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 48000)).astype("float32")
    y = core._resample_to_16k(x48, 48000)
    r["resample_len_16k"] = (len(y) == 16000)
    r["resample_preserves_rms"] = abs(np.sqrt((y**2).mean()) - np.sqrt((x48**2).mean())) < 0.02
    r["resample_passthrough"] = (len(core._resample_to_16k(x48[:16000], 16000)) == 16000)
    r["resample_empty"] = (len(core._resample_to_16k(np.zeros(0, "float32"), 48000)) == 0)
    # 44.1k -> 16k ratio
    x44 = np.zeros(44100, "float32")
    r["resample_44k"] = (abs(len(core._resample_to_16k(x44, 44100)) - 16000) <= 1)

    # --- effective_vad: pcaudio gets fast-flush, scanners unchanged ---------
    base = {"max_segment_sec": 25.0, "silence_hangover_sec": 0.8, "trigger_ratio": 3.0}
    r["vad_url_unchanged"] = (
        core.effective_vad(base, {"name": "s", "url": "x"})["max_segment_sec"] == 25.0)
    pc_vad = core.effective_vad(base, {"name": "tv", "type": "pcaudio", "device": 1})
    r["vad_pc_fast_flush"] = (pc_vad["max_segment_sec"] == 6.0)
    r["vad_pc_hangover"] = (pc_vad["silence_hangover_sec"] == 0.5)
    r["vad_pc_keeps_other"] = (pc_vad["trigger_ratio"] == 3.0)
    r["vad_override_wins"] = (core.effective_vad(
        base, {"name": "tv", "type": "pcaudio", "device": 1,
               "vad": {"max_segment_sec": 3.0}})["max_segment_sec"] == 3.0)

    # --- is_enabled semantics for pcaudio -----------------------------------
    r["pc_enabled_with_device"] = core.is_enabled({"type": "pcaudio", "device": 5}) is True
    r["pc_disabled_no_device"] = core.is_enabled({"type": "pcaudio"}) is False
    r["pc_disabled_flag"] = core.is_enabled(
        {"type": "pcaudio", "device": 5, "disabled": True}) is False
    r["url_still_works"] = core.is_enabled({"url": "http://x"}) is True
    r["url_disabled"] = core.is_enabled({"url": "http://x", "disabled": True}) is False

    # --- device listing helpers don't crash --------------------------------
    devs = core.list_input_devices()
    r["list_devices_returns_list"] = isinstance(devs, list)
    r["list_devices_shape"] = (not devs) or (len(devs[0]) == 3)

    # WDM-KS devices must be excluded (they fail at stream time, PA -9999).
    import sounddevice as sd
    ha = sd.query_hostapis()
    listed = {idx for idx, _n, _sr in devs}
    wdmks_listed = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            hn = ha[d["hostapi"]]["name"] if d.get("hostapi") is not None else ""
            if ("WDM-KS" in hn or "Kernel Streaming" in hn) and i in listed:
                wdmks_listed.append(i)
    r["wdmks_excluded"] = (wdmks_listed == [])
    # Any device find_loopback returns must be actually streamable.
    lbp = core.find_loopback_device()
    r["loopback_streamable"] = (lbp is None) or core.verify_device_streamable(lbp)
    # find_loopback returns an int index or None
    lb = core.find_loopback_device()
    r["loopback_type"] = (lb is None) or isinstance(lb, int)

    # probe helpers don't crash and return well-shaped data
    levels = core.probe_device_levels(0.05)
    r["probe_levels_list"] = isinstance(levels, list)
    r["probe_levels_shape"] = (not levels) or (len(levels[0]) == 4
                                               and isinstance(levels[0][2], float)
                                               and isinstance(levels[0][3], bool))
    r["probe_levels_sorted"] = all(
        levels[i][2] >= levels[i + 1][2] for i in range(len(levels) - 1))
    r["probe_bad_device_zero"] = (core.probe_device_level(99999, 0.05) == 0.0)

    # --- loopback vs mic classification (the webcam-mic fix) ----------------
    r["loopback_stereo_mix"] = core.is_loopback_name("Stereo Mix (Realtek(R) Audio)") is True
    r["mic_not_loopback"] = core.is_loopback_name("Microphone (Logi C270 HD WebCam)") is False
    r["whatuhear_loopback"] = core.is_loopback_name("What U Hear (Sound Blaster)") is True
    # best_loopback_by_signal must IGNORE a loud mic and pick a quieter loopback.
    # verify=False: these are synthetic indices, we're testing selection logic.
    fake = [(1, "Microphone (Webcam)", 0.02, False),   # loud mic
            (2, "Stereo Mix (Realtek)", 0.003, True),   # quieter output
            (3, "Line In", 0.05, False)]                # loud non-loopback
    r["best_ignores_mic"] = (core.best_loopback_by_signal(fake, verify=False) == 2)
    # If no loopback has signal -> None (don't fall back to a mic).
    fake2 = [(1, "Microphone", 0.02, False), (2, "Stereo Mix", 0.0, True)]
    r["best_none_when_no_loopback"] = (
        core.best_loopback_by_signal(fake2, verify=False) is None)

    # Engine.change_device mutates the stream dict and restarts the worker.
    class FakeEng:
        def __init__(s): s.added = []; s.removed = []; s.workers = {}
        # reuse real methods bound manually
    eng = core.Engine.__new__(core.Engine)
    import threading as _t
    eng._lock = _t.Lock(); eng.workers = {}; eng.stop_evt = _t.Event()
    eng.ffmpeg = ""; eng.vad_cfg = {}; eng.jobq = __import__("queue").Queue()
    eng.out = core.Output(console=False, file_logging=False)
    eng.auth_header = None; eng.player = None
    st = {"name": "PC", "type": "pcaudio", "device": 5, "color": "white"}
    eng.change_device(st, 7)
    r["change_device_mutates"] = (st["device"] == 7)
    r["change_device_started"] = ("PC" in eng.workers)
    # Changing to an output-device NAME switches to the soundcard loopback path.
    eng.stop_evt.set()
    eng.stop_evt = _t.Event(); eng.workers = {}
    st2 = {"name": "PC2", "type": "pcaudio", "device": 3, "color": "white"}
    eng.change_device(st2, "Speakers (Realtek(R) Audio)")
    r["change_device_to_name"] = (st2.get("output_device") == "Speakers (Realtek(R) Audio)"
                                  and "device" not in st2)
    r["change_device_name_worker"] = (
        isinstance(eng.workers.get("PC2"), core.LoopbackWorker))
    eng.stop_evt.set()

    # --- soundcard output-device helpers ------------------------------------
    r["soundcard_available_bool"] = isinstance(core.soundcard_available(), bool)
    outs = core.list_output_devices()
    r["list_outputs_list"] = isinstance(outs, list)
    r["list_outputs_shape"] = (not outs) or (len(outs[0]) == 2
                                             and isinstance(outs[0][1], bool))
    olevels = core.probe_output_levels(0.05)
    r["probe_outputs_shape"] = (not olevels) or (len(olevels[0]) == 3)
    r["probe_outputs_sorted"] = all(
        olevels[i][1] >= olevels[i + 1][1] for i in range(len(olevels) - 1))
    # is_enabled: pcaudio with output_device (no index) is enabled.
    r["enabled_by_output_device"] = core.is_enabled(
        {"type": "pcaudio", "output_device": "Speakers"}) is True

    # --- per-application capture (type=app) ---------------------------------
    r["proctap_available_bool"] = isinstance(core.proctap_available(), bool)
    apps = core.list_audio_apps()
    r["list_apps_list"] = isinstance(apps, list)
    r["list_apps_shape"] = (not apps) or (len(apps[0]) == 3
                                          and isinstance(apps[0][0], int)
                                          and isinstance(apps[0][2], bool))
    # is_enabled for type=app needs a pid.
    r["app_enabled_with_pid"] = core.is_enabled(
        {"type": "app", "pid": 1234}) is True
    r["app_disabled_no_pid"] = core.is_enabled({"type": "app"}) is False
    r["app_disabled_flag"] = core.is_enabled(
        {"type": "app", "pid": 1234, "disabled": True}) is False
    # app streams get the continuous-audio fast-flush too.
    r["app_fast_flush"] = (core.effective_vad(
        {"max_segment_sec": 25.0}, {"type": "app", "pid": 1})["max_segment_sec"] == 6.0)
    # ProcessLoopbackWorker constructs and exposes stop().
    import queue as _q, threading as _th
    w = core.ProcessLoopbackWorker(
        {"name": "APP", "type": "app", "pid": 1234, "app_name": "x", "color": "white"},
        {}, _q.Queue(), core.Output(console=False, file_logging=False), _th.Event())
    r["app_worker_pid"] = (w.pid == 1234)
    r["app_worker_stop"] = hasattr(w, "stop") and callable(w.stop)

    # --- DeviceWorker constructs and routes through the same jobq ------------
    import queue, threading
    jobq = queue.Queue()
    out = core.Output(console=False, file_logging=False)
    w = core.DeviceWorker({"name": "PC", "type": "pcaudio", "device": 0, "color": "white"},
                          {}, jobq, out, threading.Event())
    r["worker_name"] = (w.name_ == "PC")
    r["worker_has_stop"] = hasattr(w, "stop") and callable(w.stop)

    print("RESULTS:")
    ok = True
    for k, v in r.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "pcaudio test failed"
    print("PCAUDIO TEST: PASS")


if __name__ == "__main__":
    run()
