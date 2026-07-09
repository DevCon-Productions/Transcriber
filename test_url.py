"""Unit tests for Broadcastify URL normalization (no network)."""
from transcriber import normalize_url

cases = [
    # (input, provider, expected)
    ("https://www.broadcastify.com/listen/feed/25008", None,
     "https://audio.broadcastify.com/25008.mp3"),
    ("https://broadcastify.com/feed/11208", None,
     "https://audio.broadcastify.com/11208.mp3"),
    ("25008", "broadcastify",
     "https://audio.broadcastify.com/25008.mp3"),
    # Already-direct URL stays as-is.
    ("https://audio.broadcastify.com/25008.mp3", None,
     "https://audio.broadcastify.com/25008.mp3"),
    # Non-Broadcastify stream passes through untouched.
    ("https://npr-ice.streamguys1.com/live.mp3", None,
     "https://npr-ice.streamguys1.com/live.mp3"),
    # Bare id WITHOUT provider should NOT be treated as a feed id.
    ("25008", None, "25008"),
]

failed = 0
for url, provider, expected in cases:
    got = normalize_url(url, provider)
    ok = got == expected
    failed += not ok
    print(f"{'ok ' if ok else 'FAIL'} normalize_url({url!r}, {provider!r}) -> {got!r}")
    if not ok:
        print(f"      expected {expected!r}")

assert failed == 0, f"{failed} case(s) failed"
print("URL TEST: PASS")
