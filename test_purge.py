"""Unit test for purge_old_logs (no GPU/network)."""
import os
import time
import tempfile
import transcriber as core


def run():
    d = tempfile.mkdtemp(prefix="purgetest_")
    old = os.path.join(d, "OLD-20200101.log")
    new = os.path.join(d, "NEW-today.log")
    other = os.path.join(d, "notes.txt")  # non-.log must be left alone
    for p in (old, new, other):
        with open(p, "w", encoding="utf-8") as f:
            f.write("x\n")

    # Make 'old' 30 days old; 'new' stays current.
    thirty_days_ago = time.time() - 30 * 86400
    os.utime(old, (thirty_days_ago, thirty_days_ago))

    results = {}

    # retention_days=0 / None -> no-op
    results["disabled_zero"] = (core.purge_old_logs(0, d) == [])
    results["disabled_none"] = (core.purge_old_logs(None, d) == [])
    results["nothing_deleted_yet"] = os.path.exists(old) and os.path.exists(new)

    # retention 14 days -> deletes only the 30-day-old .log
    deleted = core.purge_old_logs(14, d)
    results["old_deleted"] = (not os.path.exists(old)) and (old in deleted)
    results["new_kept"] = os.path.exists(new)
    results["txt_kept"] = os.path.exists(other)

    # missing dir -> safe empty
    results["missing_dir_safe"] = (core.purge_old_logs(14, os.path.join(d, "nope")) == [])

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "purge test failed"
    print("PURGE TEST: PASS")


if __name__ == "__main__":
    run()
