"""Unit test for clip recording: ClipStore write/read, retention, and the
AudioPlayer's clip-over-live priority (no GPU, no network, no audio device).

Uses whatever ffmpeg find_ffmpeg() resolves, so this exercises the real Opus
encode when the build has libopus and the WAV fallback when it doesn't.
"""
import os
import tempfile
import numpy as np
import transcriber as core


def _tone(seconds=0.5, freq=440.0):
    """A float32 gate-segment-shaped array (what Gate.push returns)."""
    t = np.arange(int(core.SAMPLE_RATE * seconds), dtype=np.float32)
    return (0.4 * np.sin(2 * np.pi * freq * t / core.SAMPLE_RATE)).astype(np.float32)


def run():
    d = tempfile.mkdtemp(prefix="clips_")
    results = {}

    # ---- settings merge --------------------------------------------------
    s = core.clip_settings({})
    results["defaults_off"] = (s["enabled"] is False and s["retention_days"] == 7)
    s2 = core.clip_settings({"clips": {"enabled": True, "retention_days": 2}})
    results["settings_merge"] = (s2["enabled"] is True and s2["retention_days"] == 2
                                 and s2["bitrate"] == core.CLIP_DEFAULTS["bitrate"])

    # ---- float32 -> s16 --------------------------------------------------
    pcm = core._f32_to_s16_bytes(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))
    vals = np.frombuffer(pcm, dtype="<i2")
    results["s16_scaled"] = (len(pcm) == 10 and vals[0] == 0 and vals[1] == 32767)
    results["s16_clamped"] = (vals[3] == 32767 and vals[4] == -32767)   # no wraparound

    # ---- disabled store records nothing ----------------------------------
    off = core.ClipStore({}, clip_dir=os.path.join(d, "off"))
    off.start()
    results["disabled_saves_nothing"] = (off.save("Feed", _tone()) is None)
    off.stop()

    # ---- enabled: save -> file + index -----------------------------------
    cdir = os.path.join(d, "clips")
    cfg = {"clips": {"enabled": True, "retention_days": 7}}
    store = core.ClipStore(cfg, clip_dir=cdir)
    store.start()
    clip_id = store.save("Cleveland Fire/EMS", _tone(0.5), text="engine 14 responding")
    results["save_returns_id"] = bool(clip_id)
    # The id is minted before the file exists; the writer thread lands it after.
    store._q.join()
    path = store.path_for(clip_id)
    results["clip_on_disk"] = bool(path) and os.path.getsize(path) > 0
    results["id_is_filename_safe"] = ("/" not in clip_id)      # 'Fire/EMS' -> 'Fire-EMS'

    day = clip_id.split("-")[-3]                                # ...-YYYYMMDD-HHMMSS-N
    rows = store.index_for_day(day)
    results["index_written"] = (len(rows) == 1 and rows[0]["id"] == clip_id)
    results["index_has_text"] = (rows[0]["text"] == "engine 14 responding")
    results["index_has_feed"] = (rows[0]["feed"] == "Cleveland Fire/EMS")
    results["index_duration"] = (0.4 < rows[0]["dur"] < 0.6)

    # Opus should be dramatically smaller than the 16 KB this WAV would be.
    if path.endswith(".opus"):
        results["opus_is_small"] = (os.path.getsize(path) < 8000)
    else:
        results["opus_is_small"] = True        # no libopus in this ffmpeg build

    # ---- read back -------------------------------------------------------
    back = store.load_pcm(clip_id)
    # Opus is lossy and pads a little; just check we got roughly the right span.
    expected = int(core.SAMPLE_RATE * 0.5) * 2
    results["load_pcm_roundtrip"] = (0.7 * expected < len(back) < 1.6 * expected)
    results["load_pcm_missing"] = (store.load_pcm("no-such-clip") == b"")

    # ---- find by timestamp (re-attaching audio to a log-restored line) ----
    results["find_clip"] = (store.find_clip(rows[0]["feed"], rows[0]["ts"], day)
                            == clip_id)
    results["find_clip_miss"] = (store.find_clip("Other", rows[0]["ts"], day) is None)

    # ---- WAV fallback path -----------------------------------------------
    wdir = os.path.join(d, "wav")
    wav = core.ClipStore(cfg, clip_dir=wdir)
    wav._opus = False                       # pretend ffmpeg has no libopus
    wav.start()
    wid = wav.save("Feed", _tone(0.25))
    wav._q.join()
    wpath = wav.path_for(wid)
    results["wav_fallback"] = bool(wpath) and wpath.endswith(".wav")
    wav_pcm = wav.load_pcm(wid)
    results["wav_roundtrip_exact"] = (len(wav_pcm) == int(core.SAMPLE_RATE * 0.25) * 2)
    wav.stop()

    # ---- disk usage + purge ----------------------------------------------
    n, total = core.clips_disk_usage(cdir)
    results["usage_counts"] = (n == 1 and total > 0)
    results["purge_disabled"] = (core.purge_old_clips(0, cdir) == [])
    results["purge_keeps_new"] = (core.purge_old_clips(7, cdir) == []
                                  and store.path_for(clip_id) is not None)
    # Age the clip + its index past the window.
    import time
    old = time.time() - 30 * 86400
    for f in os.listdir(cdir):
        os.utime(os.path.join(cdir, f), (old, old))
    removed = core.purge_old_clips(7, cdir)
    results["purge_removes_old"] = (len(removed) == 2          # clip + index
                                    and store.path_for(clip_id) is None)
    results["purge_missing_dir"] = (core.purge_old_clips(7, os.path.join(d, "nope")) == [])
    store.stop()

    # ---- queue saturation drops rather than blocking ----------------------
    # A saturated queue must lose the clip, never stall the transcribe thread.
    # No writer is started here, so nothing drains what we put in.
    import queue as _queue
    full = core.ClipStore(cfg, clip_dir=os.path.join(d, "full"))
    full._thread = object()                 # make save() believe it's running
    full._q = _queue.Queue(maxsize=1)
    full._q.put(("hold", b""))              # queue is now full
    dropped = [full.save("Feed", _tone(0.05)) for _ in range(3)]
    results["queue_drops"] = (all(x is None for x in dropped) and full.dropped == 3)

    # ---- AudioPlayer: a clip preempts live audio --------------------------
    # No sound device here, so drive the callback directly with _ok forced on.
    p = core.AudioPlayer.__new__(core.AudioPlayer)
    import threading
    p._lock = threading.Lock()
    p._source = "West"
    p._stream = None
    p._sd = None
    p._buf = bytearray()
    p._clip = bytearray()
    p._ok = True

    p.feed("West", b"\x01\x02" * 100)          # live audio for the selected source
    p.feed("East", b"\x09\x09" * 100)          # other feeds are ignored
    results["feed_selected_only"] = (len(p._buf) == 200)

    out = bytearray(20)
    p._callback(out, 10, None, None)
    results["live_plays"] = (bytes(out) == b"\x01\x02" * 10)

    p.play_clip(b"\xAA\xBB" * 50)
    results["clip_clears_live"] = (len(p._buf) == 0 and p.clip_playing())
    out = bytearray(20)
    p._callback(out, 10, None, None)
    results["clip_preempts"] = (bytes(out) == b"\xAA\xBB" * 10)
    # Live audio arriving mid-clip is discarded, not queued behind it.
    p.feed("West", b"\x07\x07" * 100)
    results["live_dropped_during_clip"] = (len(p._buf) == 0)
    # Drain the rest of the clip, then live audio resumes.
    out = bytearray(80)
    p._callback(out, 40, None, None)
    results["clip_drains"] = (not p.clip_playing())
    p.feed("West", b"\x03\x04" * 10)
    out = bytearray(20)
    p._callback(out, 10, None, None)
    results["live_resumes"] = (bytes(out) == b"\x03\x04" * 10)
    # Short clip + a big ask -> the tail is zero-filled, never garbage.
    p.play_clip(b"\x05\x06" * 4)
    out = bytearray(40)
    p._callback(out, 20, None, None)
    results["clip_zero_pads"] = (bytes(out) == b"\x05\x06" * 4 + b"\x00" * 32)
    p.stop_clip()
    results["stop_clip"] = (not p.clip_playing())

    # ---- the seam: gate segment -> Transcriber -> clip + id on the line ----
    # The unit checks above test the pieces; this tests them joined up, with a
    # stub model so no Whisper/GPU is involved.
    import queue as _q2
    import threading as _th
    import time as _time

    class FakeSeg:
        def __init__(self, text):
            self.text = text
            self.no_speech_prob = 0.0
            self.avg_logprob = -0.2

    class FakeModel:
        def transcribe(self, audio, **kw):
            return [FakeSeg("engine 14 en route")], None

    edir = os.path.join(d, "e2e")
    lines = []
    out = core.Output(on_line=lambda *a: lines.append(a),
                      console=False, file_logging=False)
    estore = core.ClipStore(cfg, out=out, clip_dir=edir)
    estore.start()
    jobq, stop = _q2.Queue(), _th.Event()
    tr = core.Transcriber(FakeModel(), {}, jobq, out, stop, clips=estore,
                          should_record=lambda n: n == "West")
    tr.start()
    jobq.put(("West", "cyan", _tone(1.0), _time.time()))    # opted in
    jobq.put(("East", "red", _tone(1.0), _time.time()))     # not opted in
    jobq.join()
    estore._q.join()
    stop.set()

    results["e2e_two_lines"] = (len(lines) == 2)
    west, east = lines[0], lines[1]
    results["e2e_clip_id_delivered"] = bool(west[4])
    results["e2e_clip_file_written"] = bool(estore.path_for(west[4]))
    # Feeds that didn't opt in get a line but no clip.
    results["e2e_optout_no_clip"] = (east[4] is None)
    # The clip index timestamp must equal the line's, or a line restored from a
    # log can never be matched back to its audio.
    erow = estore.index_for_day(west[4].split("-")[-3])[0]
    results["e2e_ts_matches_line"] = (erow["ts"] == west[3])
    results["e2e_index_text"] = (erow["text"] == "engine 14 en route")
    estore.stop()

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "clip test failed"
    print("CLIPS TEST: PASS")


if __name__ == "__main__":
    run()
