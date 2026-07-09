"""Unit tests for anti-hallucination helpers (no GPU/network).

Whisper hallucinates phrases like 'Thank you.' on silence and can loop them many
times. These verify we drop pure-hallucination segments and collapse repeats
WITHOUT harming real speech."""
from transcriber import _is_hallucination, _collapse_repeats


def run():
    r = {}

    # --- _collapse_repeats: the exact reported symptom ----------------------
    fifteen = " ".join(["Thank you."] * 15)
    r["collapse_15x"] = (_collapse_repeats(fifteen) == "Thank you.")
    r["collapse_3x"] = (_collapse_repeats("Okay. Okay. Okay.") == "Okay.")
    # 2 repeats is NOT a clear loop -> keep both (conservative).
    r["keep_2x"] = (_collapse_repeats("No. No.") == "No. No.")
    # repeated single words (3+) squashed
    r["squash_words"] = (_collapse_repeats("go go go go") == "go")
    r["keep_2_words"] = (_collapse_repeats("no no") == "no no")
    # real speech is untouched
    real = "Suspect heading north on Main. Units responding. Copy that."
    r["real_untouched"] = (_collapse_repeats(real) == real)
    # mixed: a real sentence followed by a hallucination loop keeps the real part
    mixed = "Engine 14 on scene. Thank you. Thank you. Thank you."
    out = _collapse_repeats(mixed)
    r["mixed_keeps_real"] = ("Engine 14 on scene." in out and out.count("Thank you") == 1)
    r["empty_safe"] = (_collapse_repeats("") == "")

    # --- _is_hallucination --------------------------------------------------
    # Pure 'thank you' loop with high no_speech_prob -> hallucination.
    r["halluc_thankyou"] = (_is_hallucination(fifteen, 0.7) is True)
    r["halluc_single"] = (_is_hallucination("Thank you.", 0.6) is True)
    r["halluc_subscribe"] = (_is_hallucination("Please subscribe", 0.5) is True)
    # Same phrase but model WAS confident it was speech -> keep (low no_speech).
    r["confident_thankyou_kept"] = (_is_hallucination("Thank you.", 0.1) is False)
    # Real dispatch speech is never a hallucination, regardless of no_speech.
    r["real_not_halluc"] = (_is_hallucination("Adam 33 requesting backup", 0.9) is False)
    # 'thank you' embedded in a real sentence is not flagged (not pure).
    r["embedded_kept"] = (_is_hallucination("Thank you, dispatch, en route", 0.7) is False)

    print("RESULTS:")
    ok = True
    for k, v in r.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "hallucination test failed"
    print("HALLUCINATION TEST: PASS")


if __name__ == "__main__":
    run()
