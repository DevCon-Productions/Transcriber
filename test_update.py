"""Unit tests for the update-check logic (version compare + check_for_updates).
PyPI is stubbed so the test is deterministic and offline-safe."""
import transcriber as core


def run():
    results = {}

    # --- version comparison ---------------------------------------------------
    results["newer_basic"] = core.is_newer("1.2.2", "1.2.1") is True
    results["older_false"] = core.is_newer("1.2.0", "1.2.1") is False
    results["equal_false"] = core.is_newer("1.2.1", "1.2.1") is False
    results["multidigit"] = core.is_newer("1.2.10", "1.2.9") is True   # not string compare!
    results["major_bump"] = core.is_newer("2.0.0", "1.9.9") is True
    results["diff_lengths"] = core.is_newer("1.3", "1.2.5") is True
    results["none_safe"] = (core.is_newer(None, "1.2.1") is False
                            and core.is_newer("1.2.1", None) is False)

    # --- which packages are checked depends on the engine --------------------
    # (the ct2 packages aren't installed on ARM, where pywhispercpp is what runs)
    results["pkgs_ct2"] = (core.update_packages({"engine": "ct2"})
                           == ["faster-whisper", "ctranslate2"])
    results["pkgs_whispercpp"] = (core.update_packages({"engine": "whispercpp"})
                                  == ["pywhispercpp"])

    # --- check_for_updates with stubbed PyPI ---------------------------------
    # Force the ct2 engine so this is deterministic on any host (on ARM the
    # default would be pywhispercpp).
    CT2 = {"engine": "ct2"}
    fake_latest = {"faster-whisper": "9.9.9", "ctranslate2": "4.7.2"}
    core._pypi_latest = lambda pkg, timeout=4.0: fake_latest.get(pkg)
    core.installed_version = lambda pkg: {"faster-whisper": "1.2.1",
                                          "ctranslate2": "4.7.2"}.get(pkg)
    out = core.check_for_updates(cfg=CT2)
    by = {r["package"]: r for r in out}
    results["fw_update_flagged"] = (by["faster-whisper"]["update_available"] is True)
    results["ct_no_update"] = (by["ctranslate2"]["update_available"] is False)
    results["reports_versions"] = (by["faster-whisper"]["installed"] == "1.2.1"
                                   and by["faster-whisper"]["latest"] == "9.9.9")

    # An explicit package list still wins over the engine default.
    results["explicit_packages_win"] = (
        [r["package"] for r in core.check_for_updates(packages=["ctranslate2"])]
        == ["ctranslate2"])

    # On the whisper.cpp engine it checks pywhispercpp instead.
    core._pypi_latest = lambda pkg, timeout=4.0: {"pywhispercpp": "9.9.9"}.get(pkg)
    core.installed_version = lambda pkg: {"pywhispercpp": "1.5.0"}.get(pkg)
    outw = core.check_for_updates(cfg={"engine": "whispercpp"})
    results["whispercpp_checked"] = ([r["package"] for r in outw] == ["pywhispercpp"]
                                     and outw[0]["update_available"] is True)

    # --- offline: PyPI returns None -> no update, latest None ----------------
    core._pypi_latest = lambda pkg, timeout=4.0: None
    out2 = core.check_for_updates(cfg=CT2)
    results["offline_no_update"] = all(not r["update_available"] for r in out2)
    results["offline_latest_none"] = all(r["latest"] is None for r in out2)

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "update test failed"
    print("UPDATE TEST: PASS")


if __name__ == "__main__":
    run()
