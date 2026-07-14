"""Unit tests for TTS engine selection + synth backends (no audio playback).

Engine-selection logic runs anywhere; the real SAPI synth checks run only where
Windows SAPI5 voices are present (skipped otherwise).
"""
import numpy as np
import transcriber as core


def _check(r, name, cond):
    r[name] = bool(cond)


def run():
    results = {}

    # -- select_tts_engine: explicit choice wins both ways --
    _check(results, "engine_explicit_piper",
           core.select_tts_engine({"engine": "piper"}) == "piper")
    _check(results, "engine_explicit_sapi",
           core.select_tts_engine({"engine": "sapi"}) == "sapi")

    # -- auto: prefer Piper where usable, else SAPI, else piper fallback --
    op, osa = core._piper_usable, core._sapi_usable
    try:
        core._piper_usable = lambda: True
        core._sapi_usable = lambda: True
        _check(results, "auto_prefers_piper", core.select_tts_engine({}) == "piper")
        core._piper_usable = lambda: False
        _check(results, "auto_falls_back_to_sapi", core.select_tts_engine({}) == "sapi")
        core._sapi_usable = lambda: False
        _check(results, "auto_none_defaults_piper", core.select_tts_engine({}) == "piper")
    finally:
        core._piper_usable, core._sapi_usable = op, osa

    # -- engine-aware voice list + availability shape --
    vs = core.list_tts_voices(engine="sapi")
    _check(results, "voices_list_of_pairs",
           isinstance(vs, list) and all(isinstance(t, tuple) and len(t) == 2 for t in vs))
    _check(results, "tts_available_bool", isinstance(core.tts_available(), bool))

    # -- real Windows SAPI synth (skipped where unavailable) --
    if core._sapi_usable():
        _check(results, "sapi_has_voices", len(core._sapi_voices()) > 0)
        synth = core._make_tts_synth("sapi", None)
        _check(results, "sapi_engine_tag", synth.engine == "sapi")
        _check(results, "sapi_sample_rate_16k", synth.sample_rate == 16000)
        audio = synth.synthesize("Adam 12, respond code three.")
        _check(results, "sapi_pcm_int16",
               isinstance(audio, np.ndarray) and audio.dtype == np.int16 and audio.size > 1000)
        _check(results, "sapi_pcm_audible", int(np.abs(audio).max()) > 100)
        # Selecting a specific voice by description round-trips.
        name0 = core._sapi_voices()[0][0]
        _check(results, "sapi_voice_selectable",
               core._make_tts_synth("sapi", name0).voice_id == name0)
    else:
        print("  (no Windows SAPI voices on this host -> skipping SAPI synth checks)")

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "tts backend test failed"
    print("TTS BACKEND TEST: PASS")


if __name__ == "__main__":
    run()
