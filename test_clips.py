"""Unit test for clip recording: ClipStore write/read, retention, and the
AudioPlayer's clip-over-live priority (no GPU, no network, no audio device).

Uses whatever ffmpeg find_ffmpeg() resolves, so this exercises the real Opus
encode when the build has libopus and the WAV fallback when it doesn't.
"""
import json
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

    # ---- log parsing + bulk clip lookup (scrollback restore) --------------
    results["parse_log_line"] = (core.parse_log_line("[20:50:01] Copy that.")
                                 == ("20:50:01", "Copy that."))
    results["parse_log_empty_text"] = (core.parse_log_line("[20:50:01] ")
                                       == ("20:50:01", ""))
    results["parse_log_rejects"] = (core.parse_log_line("not a log line") is None
                                    and core.parse_log_line("") is None)

    mdir = os.path.join(d, "map")
    mstore = core.ClipStore(cfg, clip_dir=mdir)
    mstore.start()
    ids = [mstore.save("West", _tone(0.1), text=f"line {i}") for i in range(3)]
    mstore._q.join()
    mday = ids[0].split("-")[-3]
    cmap = mstore.clip_map(mday)
    rowsm = mstore.index_for_day(mday)
    results["clip_map_keys"] = (set(cmap) == {(r["feed"], r["ts"]) for r in rowsm})
    # Every clip is reachable: none is lost when several share a second, which
    # is exactly what these three rapid saves produce.
    results["clip_map_keeps_all"] = (
        sorted(i for ids in cmap.values() for i in ids) == sorted(ids))
    # First id per key must agree with find_clip, which it replaces in bulk.
    results["clip_map_matches_find"] = all(
        cmap[(r["feed"], r["ts"])][0] == mstore.find_clip(r["feed"], r["ts"], mday)
        for r in rowsm)
    results["clip_map_chronological"] = all(
        lst == sorted(lst) for lst in cmap.values())      # ids embed a counter
    results["clip_map_empty_day"] = (mstore.clip_map("19700101") == {})
    mstore.stop()

    # ---- exporting selected clips as one MP3 ------------------------------
    mdir2 = os.path.join(d, "mp3")
    mp3store = core.ClipStore(cfg, clip_dir=mdir2)
    mp3store.start()
    e_ids = [mp3store.save("West", _tone(0.5, freq=300 + 100 * i),
                           text=f"line {i}") for i in range(3)]
    mp3store._q.join()

    out_mp3 = os.path.join(d, "combined.mp3")
    res = core.export_clips_mp3(e_ids, out_mp3, mp3store, title="West — test")
    results["mp3_written"] = (os.path.isfile(out_mp3)
                              and os.path.getsize(out_mp3) > 0)
    results["mp3_counts_clips"] = (res["clips"] == 3 and res["missing"] == [])
    # 3 x 0.5s of audio + 2 x 300ms of silence between them.
    expected = 3 * 0.5 + 2 * (core.MP3_GAP_MS / 1000)
    results["mp3_joined_length"] = (abs(res["seconds"] - expected) < 0.15)
    # It really is an MP3, and decodes back to about that long.
    with open(out_mp3, "rb") as f:
        head = f.read(3)
    results["mp3_is_mp3"] = (head in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xfa"))
    back = mp3store.load_pcm  # reuse the ffmpeg decode path via a temp clip name
    import subprocess
    dec = subprocess.run([mp3store.ffmpeg, "-nostdin", "-loglevel", "error",
                          "-i", out_mp3, "-f", "s16le", "-ar",
                          str(core.SAMPLE_RATE), "-ac", "1", "pipe:1"],
                         capture_output=True)
    dec_sec = len(dec.stdout) / 2.0 / core.SAMPLE_RATE
    results["mp3_decodes_back"] = (abs(dec_sec - expected) < 0.35)  # codec padding

    # A single clip is a valid export too (no gap involved).
    one_mp3 = os.path.join(d, "one.mp3")
    res1 = core.export_clips_mp3([e_ids[0]], one_mp3, mp3store)
    results["mp3_single_clip"] = (res1["clips"] == 1
                                  and abs(res1["seconds"] - 0.5) < 0.1)

    # A purged clip in the selection is reported, not fatal.
    mixed = os.path.join(d, "mixed.mp3")
    res2 = core.export_clips_mp3([e_ids[0], "gone-20260101-000000-00001"],
                                 mixed, mp3store)
    results["mp3_reports_missing"] = (res2["clips"] == 1
                                      and res2["missing"] == ["gone-20260101-000000-00001"])
    # Nothing readable at all -> a clear error rather than a 0-byte file.
    try:
        core.export_clips_mp3(["nope-20260101-000000-00001"],
                              os.path.join(d, "never.mp3"), mp3store)
        results["mp3_all_missing_raises"] = False
    except RuntimeError:
        results["mp3_all_missing_raises"] = (
            not os.path.exists(os.path.join(d, "never.mp3")))

    # clip_info resolves a clip's index record from the day inside its id.
    info = mp3store.clip_info(e_ids[1])
    results["clip_info_found"] = (info is not None and info["text"] == "line 1"
                                  and info["feed"] == "West")
    results["clip_info_missing"] = (mp3store.clip_info("nope-20260101-000000-1")
                                    is None)
    mp3store.stop()

    # ---- reviewing a past day (day_summaries / load_day) ------------------
    # Two days of logs; only the newer one still has clips, which is the normal
    # state once clips.retention_days (7) outruns log_retention_days (14).
    pdir = os.path.join(d, "pastlogs")
    os.makedirs(pdir)
    with open(os.path.join(pdir, "West-20260728.log"), "w", encoding="utf-8") as f:
        f.write("[10:00:00] today one\n[10:00:00] today two\n[10:00:09] today three\n")
    with open(os.path.join(pdir, "West-20260715.log"), "w", encoding="utf-8") as f:
        f.write("[08:00:00] old one\n[08:00:01] old two\n")

    cdir2 = os.path.join(d, "pastclips")
    os.makedirs(cdir2)
    with open(os.path.join(cdir2, "index-20260728.jsonl"), "w",
              encoding="utf-8") as f:
        for cid, ts in [("c1", "10:00:00"), ("c2", "10:00:00")]:
            f.write(json.dumps({"id": cid, "feed": "West", "day": "20260728",
                                "ts": ts, "text": "x"}) + "\n")
    pstore = core.ClipStore(cfg, clip_dir=cdir2)

    summ = core.day_summaries("West", clips=pstore, log_dir=pdir)
    results["summary_newest_first"] = ([r["day"] for r in summ] ==
                                       ["20260728", "20260715"])
    results["summary_line_counts"] = ([r["lines"] for r in summ] == [3, 2])
    # The day with clips reports them; the older, purged day reports none.
    results["summary_clip_counts"] = ([r["clips"] for r in summ] == [2, 0])
    results["summary_without_store"] = (
        [r["clips"] for r in core.day_summaries("West", log_dir=pdir)] == [0, 0])

    rows = core.load_day("West", "20260728", clips=pstore, log_dir=pdir)
    results["load_day_rows"] = ([r[1] for r in rows] ==
                                ["today one", "today two", "today three"])
    # Same-second lines take different clips, in order; the third has none.
    results["load_day_distinct_clips"] = ([r[2] for r in rows] ==
                                          ["c1", "c2", None])
    old_rows = core.load_day("West", "20260715", clips=pstore, log_dir=pdir)
    results["load_day_text_only"] = (len(old_rows) == 2
                                     and all(r[2] is None for r in old_rows))
    results["load_day_missing"] = (core.load_day("West", "19700101",
                                                 clips=pstore, log_dir=pdir) == [])
    results["load_day_unknown_feed"] = (core.load_day("Nope", "20260728",
                                                      clips=pstore,
                                                      log_dir=pdir) == [])

    # ---- upgrade path: a config written before clips existed ---------------
    # 2.0 added clip recording, but an EXISTING config is never overwritten on
    # upgrade (deliberate -- feeds must survive). That left upgraded installs with
    # no "clips" block at all: recording fell back to defaults (off), the per-feed
    # checkboxes did nothing, and nothing in the UI said why. load_config now
    # fills in absent blocks so the feature is merely OFF, not invisible.
    updir = os.path.join(d, "upgrade")
    os.makedirs(updir)
    old_cfg = os.path.join(updir, "config.json")
    with open(old_cfg, "w", encoding="utf-8") as f:
        json.dump({"model": "base.en", "log_retention_days": 14,
                   "streams": [{"name": "A", "url": "u", "record": True}]}, f)
    up = core.load_config(old_cfg)
    results["upgrade_adds_block"] = (up.get("clips") == core.CLIP_DEFAULTS)
    results["upgrade_persists"] = (
        "clips" in json.load(open(old_cfg, encoding="utf-8")))
    # Filling the gap must not silently switch recording ON.
    results["upgrade_stays_off"] = (up["clips"]["enabled"] is False)
    results["upgrade_keeps_rest"] = (up["model"] == "base.en"
                                     and up["streams"][0]["record"] is True)
    # An existing block is authoritative -- never overwritten or re-defaulted.
    mine = os.path.join(updir, "mine.json")
    with open(mine, "w", encoding="utf-8") as f:
        json.dump({"model": "x", "clips": {"enabled": True, "retention_days": 3}}, f)
    got = core.load_config(mine)
    results["upgrade_respects_existing"] = (
        got["clips"] == {"enabled": True, "retention_days": 3})
    results["upgrade_no_needless_write"] = (
        json.load(open(mine, encoding="utf-8"))["clips"]
        == {"enabled": True, "retention_days": 3})
    # Per-key defaults still fill in at read time for a partial block.
    results["upgrade_partial_merged"] = (core.clip_settings(got)["bitrate"]
                                         == core.CLIP_DEFAULTS["bitrate"])
    # Idempotent: loading again changes nothing.
    before = open(old_cfg, encoding="utf-8").read()
    core.load_config(old_cfg)
    results["upgrade_idempotent"] = (open(old_cfg, encoding="utf-8").read() == before)

    # ---- retention: size cap and the combined policy -----------------------
    import time as _t2
    sdir = os.path.join(d, "sizecap")
    os.makedirs(sdir)

    def _seed(n_files, mb, age_days_start):
        """n_files clips of mb MB each, ages counting DOWN from age_days_start."""
        made = []
        for i in range(n_files):
            p = os.path.join(sdir, f"West-2026070{i}-000000-{i:05d}.opus")
            with open(p, "wb") as f:
                f.write(b"x" * int(mb * (1 << 20)))
            when = _t2.time() - (age_days_start - i) * 86400
            os.utime(p, (when, when))
            made.append(p)
        return made

    _seed(10, 1, 10)                     # 10 MB total, ages 10d..1d
    results["cap_usage_before"] = (core.clips_disk_usage(sdir) == (10, 10 << 20))
    evicted = core.purge_clips_over_size(4 << 20, sdir)
    surviving = sorted(os.listdir(sdir))
    results["cap_evicts_to_fit"] = (core.clips_disk_usage(sdir)[1] <= (4 << 20))
    # It must take the OLDEST, not an arbitrary set.
    results["cap_evicts_oldest"] = (len(evicted) == 6
                                    and surviving == [f"West-2026070{i}-000000-"
                                                      f"{i:05d}.opus"
                                                      for i in range(6, 10)])
    results["cap_disabled_noop"] = (core.purge_clips_over_size(0, sdir) == [])
    results["cap_under_noop"] = (core.purge_clips_over_size(1 << 30, sdir) == [])
    results["cap_missing_dir"] = (
        core.purge_clips_over_size(1, os.path.join(d, "nope")) == [])

    # Combined: age first, then the cap on what's left. Ordering matters --
    # expiring by age first means the cap evicts as few live clips as possible.
    cdir3 = os.path.join(d, "combined")
    os.makedirs(cdir3)
    sdir_save = sdir
    sdir = cdir3
    _seed(6, 1, 30)                      # ages 30d..25d -> all past a 7-day window
    _seed(4, 1, 4)                       # overwrites 4 of them at ages 4d..1d
    sdir = sdir_save
    aged, over = core.apply_clip_retention(
        {"retention_days": 7, "max_gb": 2 / 1024.0}, cdir3)     # 2 MB cap
    results["combined_ages_out"] = (len(aged) >= 1)
    results["combined_then_caps"] = (core.clips_disk_usage(cdir3)[1] <= (2 << 20))
    results["combined_reports_both"] = (isinstance(aged, list)
                                        and isinstance(over, list))
    # No cap set -> only the age policy runs.
    cdir4 = os.path.join(d, "agesonly")
    os.makedirs(cdir4)
    sdir_save2, sdir = sdir, cdir4
    _seed(3, 1, 2)                       # all recent
    sdir = sdir_save2
    aged2, over2 = core.apply_clip_retention({"retention_days": 7, "max_gb": 0},
                                             cdir4)
    results["combined_no_cap"] = (aged2 == [] and over2 == []
                                  and core.clips_disk_usage(cdir4)[0] == 3)

    # Settings default the cap off, so existing configs behave exactly as before.
    results["cap_default_off"] = (core.clip_settings({})["max_gb"] == 0)
    results["cap_from_cfg"] = (
        core.clip_settings({"clips": {"max_gb": 3}})["max_gb"] == 3)

    # Log usage readout (the retention dialog shows both side by side).
    ldir2 = os.path.join(d, "logsize")
    os.makedirs(ldir2)
    with open(os.path.join(ldir2, "West-20260728.log"), "w", encoding="utf-8") as f:
        f.write("[00:00:00] hello\n")
    results["logs_usage"] = (core.logs_disk_usage(ldir2)[0] == 1
                             and core.logs_disk_usage(ldir2)[1] > 0)

    # ---- transcript bundles (.tscript) ------------------------------------
    import zipfile
    bdir = os.path.join(d, "bundle")
    bstore = core.ClipStore(cfg, clip_dir=bdir)
    bstore.start()
    b_ids = [bstore.save("West", _tone(0.4), text=f"bundle line {i}")
             for i in range(3)]
    bstore._q.join()
    b_rows = [("09:00:00", "bundle line 0", b_ids[0]),
              ("09:00:01", "bundle line 1", b_ids[1]),
              ("09:00:02", "no audio here", None),          # kept, just silent
              ("09:00:03", "bundle line 2", b_ids[2])]

    bpath = os.path.join(d, "incident" + core.TRANSCRIPT_EXT)
    wres = core.write_transcript_bundle(bpath, "West", b_rows, bstore,
                                        app_version="test", day="20260728")
    results["bundle_written"] = (os.path.isfile(bpath) and wres["lines"] == 4
                                 and wres["clips"] == 3 and wres["missing"] == [])

    # It must be an ordinary ZIP anyone can open -- that's the whole argument
    # for not inventing a private container.
    with zipfile.ZipFile(bpath) as z:
        names = z.namelist()
        man = json.loads(z.read("manifest.json"))
        tr = json.loads(z.read("transcript.json"))
    results["bundle_is_zip"] = ("manifest.json" in names
                                and "transcript.json" in names)
    results["bundle_carries_clips"] = (
        sum(1 for n in names if n.startswith("clips/")) == 3)
    results["bundle_manifest"] = (man["format"] == core.TRANSCRIPT_FORMAT
                                  and man["feed"] == "West"
                                  and man["lines"] == 4 and man["clips"] == 3
                                  and man["first"] == "09:00:00"
                                  and man["last"] == "09:00:03")
    results["bundle_text_readable"] = ([e["text"] for e in tr] ==
                                       [r[1] for r in b_rows])

    # Reading it back gives the same rows the live views use.
    with core.TranscriptBundle(bpath) as b:
        results["bundle_rows"] = (len(b.rows) == 4
                                  and b.feed == "West" and b.day == "20260728")
        results["bundle_keeps_clipless"] = (b.rows[2][2] is None
                                            and b.rows[2][1] == "no audio here")
        results["bundle_clip_ids"] = ([r[2] for r in b.rows] ==
                                      [b_ids[0], b_ids[1], None, b_ids[2]])
        # Audio decodes straight from inside the archive, no extraction.
        pcm = b.load_pcm(b_ids[0])
        results["bundle_plays_from_zip"] = (
            abs(len(pcm) / 2 / core.SAMPLE_RATE - 0.4) < 0.15)
        results["bundle_missing_clip"] = (b.load_pcm("nope") == b"")
        results["bundle_clip_info"] = (
            (b.clip_info(b_ids[1]) or {}).get("text") == "bundle line 1")
        # MP3 export takes a bundle as its source unchanged.
        bmp3 = os.path.join(d, "from_bundle.mp3")
        mres = core.export_clips_mp3([b_ids[0], b_ids[2]], bmp3, b)
        results["bundle_feeds_mp3"] = (mres["clips"] == 2
                                       and os.path.getsize(bmp3) > 0)

    # The audio survives the originals being purged -- the reason to save one.
    for f in os.listdir(bdir):
        os.remove(os.path.join(bdir, f))
    results["bundle_outlives_purge"] = (bstore.load_pcm(b_ids[0]) == b"")
    with core.TranscriptBundle(bpath) as b:
        results["bundle_still_plays"] = (len(b.load_pcm(b_ids[0])) > 0)
    bstore.stop()

    # Malformed inputs fail with a clear message, never a traceback.
    notzip = os.path.join(d, "notzip" + core.TRANSCRIPT_EXT)
    with open(notzip, "w", encoding="utf-8") as f:
        f.write("this is just text")
    try:
        core.TranscriptBundle(notzip)
        results["bundle_rejects_nonzip"] = False
    except ValueError:
        results["bundle_rejects_nonzip"] = True

    wrongzip = os.path.join(d, "wrong.zip")
    with zipfile.ZipFile(wrongzip, "w") as z:
        z.writestr("hello.txt", "not a transcript")
    try:
        core.TranscriptBundle(wrongzip)
        results["bundle_rejects_other_zip"] = False
    except ValueError:
        results["bundle_rejects_other_zip"] = True

    futurezip = os.path.join(d, "future" + core.TRANSCRIPT_EXT)
    with zipfile.ZipFile(futurezip, "w") as z:
        z.writestr("manifest.json", json.dumps({
            "format": core.TRANSCRIPT_FORMAT,
            "version": core.TRANSCRIPT_VERSION + 5, "feed": "X"}))
        z.writestr("transcript.json", "[]")
    try:
        core.TranscriptBundle(futurezip)
        results["bundle_rejects_future"] = False
    except ValueError as e:
        # Must say WHY, so the user knows to update rather than assume damage.
        results["bundle_rejects_future"] = ("newer version" in str(e))

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
