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

    # -- select_tts_engine: explicit choice wins --
    _check(results, "engine_explicit_piper",
           core.select_tts_engine({"engine": "piper"}) == "piper")
    _check(results, "engine_explicit_sapi",
           core.select_tts_engine({"engine": "sapi"}) == "sapi")
    _check(results, "engine_explicit_winrt",
           core.select_tts_engine({"engine": "winrt"}) == "winrt")

    # -- auto: piper (if usable) -> winrt -> sapi -> piper fallback --
    op, ow, osa = core._piper_usable, core._winrt_usable, core._sapi_usable
    try:
        core._piper_usable = lambda: True
        core._winrt_usable = lambda: True
        core._sapi_usable = lambda: True
        _check(results, "auto_prefers_piper", core.select_tts_engine({}) == "piper")
        core._piper_usable = lambda: False
        _check(results, "auto_then_winrt", core.select_tts_engine({}) == "winrt")
        core._winrt_usable = lambda: False
        _check(results, "auto_then_sapi", core.select_tts_engine({}) == "sapi")
        core._sapi_usable = lambda: False
        _check(results, "auto_none_defaults_piper", core.select_tts_engine({}) == "piper")
    finally:
        core._piper_usable, core._winrt_usable, core._sapi_usable = op, ow, osa

    # -- cross-engine voice matching (a name saved under one engine resolves
    #    under another; SAPI 'Microsoft Zira Desktop - ...' -> WinRT 'Microsoft Zira')
    names = ["Microsoft David", "Microsoft Zira", "Microsoft Mark"]
    _check(results, "match_exact", core._match_voice_name("Microsoft Mark", names) == "Microsoft Mark")
    _check(results, "match_case_insensitive",
           core._match_voice_name("microsoft zira", names) == "Microsoft Zira")
    _check(results, "match_cross_engine_prefix",
           core._match_voice_name("Microsoft Zira Desktop - English (United States)",
                                  names) == "Microsoft Zira")
    _check(results, "match_unknown_none",
           core._match_voice_name("No Such Voice 9000", names) is None)
    _check(results, "match_empty_safe",
           core._match_voice_name(None, names) is None
           and core._match_voice_name("Microsoft Mark", []) is None)

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

    # -- real WinRT/OneCore synth (skipped where unavailable) --
    if core._winrt_usable():
        wv = core._winrt_voices()
        _check(results, "winrt_has_voices", len(wv) > 0)
        # WinRT sees every installed voice: should be >= what classic SAPI sees.
        _check(results, "winrt_superset_of_sapi", len(wv) >= len(core._sapi_voices()))
        synth = core._make_tts_synth("winrt", None)
        _check(results, "winrt_engine_tag", synth.engine == "winrt")
        audio = synth.synthesize("Adam 12, respond code three.")
        _check(results, "winrt_pcm_int16",
               isinstance(audio, np.ndarray) and audio.dtype == np.int16 and audio.size > 1000)
        _check(results, "winrt_pcm_audible", int(np.abs(audio).max()) > 100)
        _check(results, "winrt_sample_rate_sane", 8000 <= synth.sample_rate <= 48000)
        # Selecting a specific voice by display name round-trips.
        name0 = wv[0][0]
        _check(results, "winrt_voice_selectable",
               core._make_tts_synth("winrt", name0).voice_id == name0)
        # An unknown voice name falls back to the default rather than raising.
        _check(results, "winrt_unknown_voice_falls_back",
               core._make_tts_synth("winrt", "No Such Voice 9000").voice_id in
               [n for n, _ in wv])
    else:
        print("  (no WinRT voices on this host -> skipping WinRT synth checks)")

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "tts backend test failed"
    print("TTS BACKEND TEST: PASS")


if __name__ == "__main__":
    run()
