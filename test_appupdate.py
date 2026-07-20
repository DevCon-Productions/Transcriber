"""Tests for the GitHub-release app self-updater (no real network)."""
import io, json, os, tempfile, urllib.request
import transcriber as core


class _Resp(io.BytesIO):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}
    def __enter__(self): return self
    def __exit__(self, *a): self.close()
    def get(self, k, d=None): return self.headers.get(k, d)


def _release_json(tag, assets):
    return json.dumps({
        "tag_name": tag, "html_url": f"https://x/releases/tag/{tag}",
        "body": "notes here", "assets": assets,
    }).encode()


def run():
    r = {}
    orig = urllib.request.urlopen

    def fake(url_or_req, timeout=None):
        return _Resp(_release_json("v1.5",
            [{"name": "Transcriber-Setup-1.5.exe",
              "browser_download_url": "https://x/dl/Transcriber-Setup-1.5.exe",
              "size": 511 * (1 << 20)}]))
    try:
        # 1. newer version available + asset parsed
        urllib.request.urlopen = fake
        info = core.check_for_app_update("1.2")
        r["available_true"] = (info is not None and info["available"] is True)
        r["latest_parsed"] = (info["latest"] == "1.5")            # 'v' stripped
        r["asset_url"] = (info["asset_url"].endswith("1.5.exe"))
        r["asset_size"] = (info["asset_size"] == 511 * (1 << 20))
        r["notes"] = (info["notes"] == "notes here")

        # 2. same/older running version -> not available
        info2 = core.check_for_app_update("1.5")
        r["same_not_available"] = (info2["available"] is False)
        info3 = core.check_for_app_update("2.0")
        r["older_remote_not_available"] = (info3["available"] is False)

        # 3. release with no .exe asset -> asset_url None, available still computed
        urllib.request.urlopen = lambda u, timeout=None: _Resp(
            _release_json("v1.6", [{"name": "notes.txt",
                                    "browser_download_url": "https://x/notes.txt"}]))
        info4 = core.check_for_app_update("1.2")
        r["no_asset_url_none"] = (info4["asset_url"] is None
                                  and info4["available"] is True)

        # 4. network failure -> None (silent)
        def boom(u, timeout=None): raise OSError("no network")
        urllib.request.urlopen = boom
        r["network_error_none"] = (core.check_for_app_update("1.2") is None)

        # 4b. architecture-aware asset picker (shared release, both installers).
        both = [
            {"name": "Transcriber-Setup-1.5.exe",
             "browser_download_url": "https://x/dl/x64.exe", "size": 1},
            {"name": "Transcriber-ARM64-Setup-1.5.exe",
             "browser_download_url": "https://x/dl/arm.exe", "size": 2},
        ]
        urllib.request.urlopen = lambda u, timeout=None: _Resp(
            _release_json("v1.5", both))
        _orig_arm = core.interpreter_is_arm64
        try:
            # running as x64 -> must pick the NON-arm installer
            core.interpreter_is_arm64 = lambda: False
            ix = core.check_for_app_update("1.2")
            r["x64_picks_x64_asset"] = (ix["asset_name"] == "Transcriber-Setup-1.5.exe")
            # running as arm64 -> must pick the arm installer
            core.interpreter_is_arm64 = lambda: True
            ia = core.check_for_app_update("1.2")
            r["arm_picks_arm_asset"] = (ia["asset_name"] == "Transcriber-ARM64-Setup-1.5.exe")
            # x64 app, release has ONLY an arm asset -> no cross-install (asset None)
            core.interpreter_is_arm64 = lambda: False
            urllib.request.urlopen = lambda u, timeout=None: _Resp(
                _release_json("v1.5", [both[1]]))
            ionly = core.check_for_app_update("1.2")
            r["x64_refuses_arm_only"] = (ionly["available"] is True
                                         and ionly["asset_url"] is None)
        finally:
            core.interpreter_is_arm64 = _orig_arm
        # picker returns None on empty / no matching arch
        r["picker_empty_none"] = (core._pick_installer_asset([]) is None)

        # 5. download_file writes dest, fires progress, cleans .part
        payload = b"x" * (3 * (1 << 20) + 123)
        urllib.request.urlopen = lambda u, timeout=None: _Resp(
            payload, {"Content-Length": str(len(payload))})
        tmp = tempfile.mkdtemp()
        dest = os.path.join(tmp, "sub", "Setup.exe")
        seen = []
        out = core.download_file("https://x/dl", dest,
                                 progress_cb=lambda d, t: seen.append((d, t)))
        r["dl_returns_dest"] = (out == dest)
        r["dl_wrote_all_bytes"] = (os.path.getsize(dest) == len(payload))
        r["dl_progress_fired"] = (len(seen) > 0 and seen[-1][0] == len(payload))
        r["dl_total_reported"] = (seen[-1][1] == len(payload))
        r["dl_no_part_left"] = (not os.path.exists(dest + ".part"))
    finally:
        urllib.request.urlopen = orig

    print("RESULTS:")
    ok = True
    for k, v in r.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "appupdate test failed"
    print("APPUPDATE TEST: PASS")


if __name__ == "__main__":
    run()
