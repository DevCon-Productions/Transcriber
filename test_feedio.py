"""Unit test for feed-list export/import and transcript->PDF (no GPU/network).

Covers the portability rules that matter when moving a list between the x64 and
ARM64 installs (which keep separate data dirs): engine settings never travel,
session-only fields are dropped, and the reader accepts a raw config.json too.
"""
import json
import os
import re
import tempfile
import transcriber as core


def _pdf_objects_resolve(data):
    """Every xref offset must point at the '<n> 0 obj' it claims, and every
    stream's /Length must land exactly on 'endstream'. This is what a reader
    checks first, so a pass here means the file opens."""
    start = int(data[data.rindex(b"startxref"):].split()[1])
    if data[start:start + 4] != b"xref":
        return False
    rows = data[start:].split(b"\n")
    count = int(rows[1].split()[1])
    for n in range(1, count):
        off = int(rows[2 + n].split()[0])
        if not data[off:off + 16].startswith(b"%d 0 obj" % n):
            return False
    for m in re.finditer(rb"<< /Length (\d+) >>\nstream\n", data):
        end = m.end() + int(m.group(1))
        if data[end:end + 10] != b"\nendstream":
            return False
    return True


def run():
    d = tempfile.mkdtemp(prefix="feedio_")
    results = {}

    library = [
        {"name": "Cleveland West", "url": "https://www.broadcastify.com/listen/feed/25008",
         "color": "cyan", "provider": "broadcastify", "location": "Cleveland, OH",
         "desc": "PD West"},
        {"name": "TV Room", "type": "pcaudio", "output_device": "Speakers (Realtek)",
         "color": "green", "disabled": True},
        {"name": "Browser", "type": "app", "pid": 4242, "app_name": "chrome.exe",
         "color": "red"},
    ]

    # ---- export ----------------------------------------------------------
    path = os.path.join(d, "feeds.json")
    n = core.export_feeds(path, library, app_version="test")
    doc = json.load(open(path, encoding="utf-8"))
    results["export_count"] = (n == 3 and len(doc["feeds"]) == 3)
    results["export_format"] = (doc["format"] == core.FEED_EXPORT_FORMAT
                                and doc["version"] == core.FEED_EXPORT_VERSION)
    # Engine settings must NOT ride along: an ARM box importing 'large-v3'/cuda
    # would try a 3 GB download for a model it can't run.
    results["no_engine_keys"] = not any(
        k in doc for k in ("model", "device", "compute_type", "engine"))
    # Session-only fields are dropped; portable ones survive.
    results["pid_dropped"] = all("pid" not in f for f in doc["feeds"])
    results["disabled_dropped"] = all("disabled" not in f for f in doc["feeds"])
    results["url_kept"] = doc["feeds"][0]["url"].endswith("/25008")
    results["location_kept"] = doc["feeds"][0]["location"] == "Cleveland, OH"
    results["type_kept"] = doc["feeds"][1]["type"] == "pcaudio"

    # ---- import: our own file --------------------------------------------
    feeds, warnings = core.import_feeds(path)
    results["roundtrip_count"] = (len(feeds) == 3)
    results["roundtrip_names"] = ([f["name"] for f in feeds] ==
                                  [e["name"] for e in library])
    # The app-capture feed is machine-bound either way -> must be flagged.
    results["app_warned"] = any("Browser" in w for w in warnings)

    # ---- import: a bare JSON array ---------------------------------------
    p_bare = os.path.join(d, "bare.json")
    json.dump([{"name": "X", "url": "u"}, {"no_name": 1}], open(p_bare, "w"))
    feeds_bare, _ = core.import_feeds(p_bare)
    results["bare_array"] = ([f["name"] for f in feeds_bare] == ["X"])

    # ---- import: a whole config.json from the other install ---------------
    p_cfg = os.path.join(d, "config.json")
    json.dump({"model": "large-v3", "device": "cuda",
               "feed_library": [{"name": "A", "url": "a"}],
               "streams": [{"name": "B", "url": "b", "disabled": False},
                           {"name": "A", "url": "a"}]},         # dupe of library
              open(p_cfg, "w"))
    feeds_cfg, _ = core.import_feeds(p_cfg)
    results["config_json"] = ([f["name"] for f in feeds_cfg] == ["A", "B"])

    # ---- import: rejects junk --------------------------------------------
    p_bad = os.path.join(d, "bad.json")
    json.dump({"hello": 1}, open(p_bad, "w"))
    try:
        core.import_feeds(p_bad)
        results["bad_rejected"] = False
    except ValueError:
        results["bad_rejected"] = True

    # ---- PDF -------------------------------------------------------------
    lines = [f"[0{i // 10}:0{i % 10}:00] unit {i} responding to a call " + "w" * (i % 70)
             for i in range(90)]
    lines.append('[00:00:00] smart “quotes”, (parens), back\\slash, '
                 '— dash, café')
    lines.append("[00:00:01] 中文 outside latin-1")
    out = os.path.join(d, "t.pdf")
    pages = core.write_transcript_pdf(out, "Cleveland Fire/EMS", lines,
                                      subtitle="2026-07-12 – 2026-07-19")
    data = open(out, "rb").read()
    results["pdf_header"] = data.startswith(b"%PDF-1.4")
    results["pdf_multipage"] = pages >= 2
    results["pdf_structure"] = _pdf_objects_resolve(data)
    results["pdf_page_count"] = (data.count(b"/Type /Page ") == pages)
    # Parens/backslash escaped, not emitted raw (they'd break the content stream).
    results["pdf_escapes"] = (b"\\(parens\\)" in data and b"back\\\\slash" in data)
    # Non-latin-1 degrades to '?' instead of corrupting the stream.
    results["pdf_unicode_safe"] = b"?? outside latin-1" in data
    results["pdf_footer"] = b"Page 1 of " in data
    # A feed with no lines still produces a valid one-page file, not a crash.
    empty = os.path.join(d, "empty.pdf")
    results["pdf_empty_ok"] = (core.write_transcript_pdf(empty, "Nothing", []) == 1
                               and _pdf_objects_resolve(open(empty, "rb").read()))

    # ---- log discovery ---------------------------------------------------
    logs = os.path.join(d, "logs")
    os.makedirs(logs)
    for day in ("20260719", "20260712", "20260715"):
        with open(os.path.join(logs, f"Cleveland Fire-EMS-{day}.log"), "w",
                  encoding="utf-8") as f:
            f.write(f"[00:00:00] {day} line\n")
    with open(os.path.join(logs, "Other Feed-20260719.log"), "w",
              encoding="utf-8") as f:
        f.write("[00:00:00] not ours\n")
    found = core.log_files_for("Cleveland Fire/EMS", log_dir=logs)  # '/' -> '-'
    results["logs_matched"] = ([day for day, _p in found] ==
                               ["20260712", "20260715", "20260719"])   # oldest first
    read = core.read_log_lines([p for _d, p in found] + [os.path.join(logs, "gone.log")])
    results["logs_read"] = (len(read) == 3 and read[0].endswith("20260712 line"))

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "feed export/import + PDF test failed"
    print("FEED IO TEST: PASS")


if __name__ == "__main__":
    run()
