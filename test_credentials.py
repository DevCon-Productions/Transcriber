"""Tests for Broadcastify credential save/load + placeholder handling."""
import os, tempfile, json
import transcriber as core


def run():
    r = {}
    # Redirect CREDENTIALS_PATH to a temp file for the duration.
    tmp = tempfile.mkdtemp()
    orig = core.CREDENTIALS_PATH
    core.CREDENTIALS_PATH = os.path.join(tmp, "credentials.json")
    # ensure no env creds interfere
    for k in ("BROADCASTIFY_USERNAME", "BROADCASTIFY_PASSWORD"):
        os.environ.pop(k, None)
    try:
        # 1. no file -> not configured
        r["none_when_missing"] = (core.load_credentials() == (None, None))
        r["not_configured_missing"] = (core.credentials_configured() is False)

        # 2. placeholder file -> treated as not configured
        with open(core.CREDENTIALS_PATH, "w", encoding="utf-8") as f:
            json.dump({"broadcastify": {"username": "YOUR_BROADCASTIFY_USERNAME",
                                        "password": "YOUR_BROADCASTIFY_PASSWORD"}}, f)
        r["placeholder_ignored"] = (core.load_credentials() == (None, None))
        r["placeholder_not_configured"] = (core.credentials_configured() is False)

        # 3. save + load round-trip
        ok = core.save_credentials("testuser23", "s3cr3t&P@ss")
        r["save_ok"] = (ok is True)
        r["load_roundtrip"] = (core.load_credentials() == ("testuser23", "s3cr3t&P@ss"))
        r["configured_after_save"] = (core.credentials_configured() is True)

        # 4. blank values ignored
        core.save_credentials("", "")
        r["blank_not_configured"] = (core.credentials_configured() is False)

        # 5. save preserves other top-level keys
        with open(core.CREDENTIALS_PATH, "w", encoding="utf-8") as f:
            json.dump({"broadcastify": {"username": "a", "password": "b"},
                       "other": {"keep": 1}}, f)
        core.save_credentials("newuser", "newpass")
        with open(core.CREDENTIALS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        r["preserves_other_keys"] = (data.get("other") == {"keep": 1})
        r["updates_broadcastify"] = (data["broadcastify"]["username"] == "newuser")

        # 6. broadcastify stream detection
        r["detect_by_url"] = core.is_broadcastify_stream(
            {"url": "https://audio.broadcastify.com/25008.mp3"})
        r["detect_by_provider"] = core.is_broadcastify_stream(
            {"provider": "broadcastify", "url": "x"})
        r["detect_negative_pc"] = (core.is_broadcastify_stream(
            {"type": "pcaudio", "output_device": "Realtek"}) is False)
        r["detect_negative_other_url"] = (core.is_broadcastify_stream(
            {"url": "http://example.com/stream.mp3"}) is False)

        # 7. Engine.apply_credentials: saves, rebuilds auth, restarts ONLY the
        #    running Broadcastify feeds (leaves pc-audio / other feeds alone).
        core.save_credentials("olduser", "oldpass")
        eng = core.Engine({"streams": []}, console=False, file_logging=False)
        calls = {"removed": [], "added": []}
        eng.remove_stream = lambda n: calls["removed"].append(n)
        eng.add_stream = lambda s: calls["added"].append(s["name"])
        eng.stream_names = lambda: ["Cleveland West", "TV Audio", "Off Feed"]
        active = [
            {"name": "Cleveland West", "url": "https://audio.broadcastify.com/25008.mp3"},
            {"name": "TV Audio", "type": "pcaudio", "output_device": "Realtek"},
            # "Off Feed" is a bcfy feed but NOT in active list -> must be skipped
        ]
        saved = eng.apply_credentials("freshuser", "freshpass", active)
        r["apply_saved"] = (saved is True)
        r["apply_persisted"] = (core.load_credentials() == ("freshuser", "freshpass"))
        r["apply_auth_rebuilt"] = (eng.auth_header is not None)
        r["apply_restarts_bcfy"] = (calls["removed"] == ["Cleveland West"]
                                    and calls["added"] == ["Cleveland West"])
        r["apply_skips_pcaudio"] = ("TV Audio" not in calls["removed"])
        r["apply_skips_unknown"] = ("Off Feed" not in calls["removed"])
    finally:
        core.CREDENTIALS_PATH = orig

    print("RESULTS:")
    ok = True
    for k, v in r.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "credentials test failed"
    print("CREDENTIALS TEST: PASS")


if __name__ == "__main__":
    run()
