"""Unit tests for the transcription-backend abstraction (no compiler / model needed).

Covers: backend selection (explicit engine + arch-based), the arch probe keying off
PROCESSOR_ARCHITECTURE (not platform.machine()), faster-whisper -> GGML model-name
mapping, the pywhispercpp -> faster-whisper segment-shape adapter, and the
WhisperCppBackend wiring/kwarg mapping via an injected fake model.
"""
import os
import math
import numpy as np
import transcriber as core


class FakeSeg:
    """Stand-in for a pywhispercpp Segment (only fields the adapter reads)."""
    def __init__(self, text, probability):
        self.text = text
        self.probability = probability


class FakeWCModel:
    """Stand-in for pywhispercpp.model.Model: records the transcribe kwargs and
    returns canned segments, so we can assert the faster-whisper -> whisper.cpp
    parameter mapping without a compiled backend or a real model file."""
    def __init__(self, segs):
        self._segs = segs
        self.last_kwargs = None
        self.last_audio = None

    def transcribe(self, audio, **kw):
        self.last_audio = audio
        self.last_kwargs = kw
        return self._segs


def _check(results, name, cond):
    results[name] = bool(cond)


def run():
    results = {}

    # -- select_backend: explicit engine wins, both directions + aliases --
    _check(results, "engine_ct2", core.select_backend({"engine": "ct2"}) == "ct2")
    _check(results, "engine_whispercpp",
           core.select_backend({"engine": "whispercpp"}) == "whispercpp")
    _check(results, "engine_alias_faster_whisper",
           core.select_backend({"engine": "faster-whisper"}) == "ct2")
    _check(results, "engine_alias_dotted",
           core.select_backend({"engine": "whisper.cpp"}) == "whispercpp")

    # -- select_backend: arch-based default (monkeypatch the arch probe) --
    orig = core.interpreter_is_arm64
    try:
        core.interpreter_is_arm64 = lambda: True
        _check(results, "arm_defaults_whispercpp", core.select_backend({}) == "whispercpp")
        core.interpreter_is_arm64 = lambda: False
        _check(results, "x64_defaults_ct2", core.select_backend({}) == "ct2")
    finally:
        core.interpreter_is_arm64 = orig

    # -- interpreter_is_arm64 keys off PROCESSOR_ARCHITECTURE, NOT machine() --
    saved = os.environ.get("PROCESSOR_ARCHITECTURE")
    try:
        os.environ["PROCESSOR_ARCHITECTURE"] = "AMD64"
        _check(results, "amd64_env_not_arm", core.interpreter_is_arm64() is False)
        os.environ["PROCESSOR_ARCHITECTURE"] = "ARM64"
        _check(results, "arm64_env_is_arm", core.interpreter_is_arm64() is True)
    finally:
        if saved is None:
            os.environ.pop("PROCESSOR_ARCHITECTURE", None)
        else:
            os.environ["PROCESSOR_ARCHITECTURE"] = saved

    # -- model-name mapping (tolerant of pywhispercpp being absent) --
    _check(results, "map_small_en_passthrough",
           core.whispercpp_model_name("small.en") == "small.en")
    _check(results, "map_distil_to_ggml",
           core.whispercpp_model_name("distil-large-v3") in
           ("large-v3-turbo", core.ARM_DEFAULT_MODEL, "small.en"))

    # -- segment adapter: probability -> (no_speech_prob, avg_logprob) --
    segs = core._map_whispercpp_segments([
        FakeSeg(" hello", 1.0),             # perfect confidence
        FakeSeg(" maybe", 0.5),
        FakeSeg(" unknown", float("nan")),  # no confidence number -> neutral, kept
    ])
    _check(results, "adapter_count", len(segs) == 3)
    _check(results, "adapter_text_preserved", segs[0].text == " hello")
    _check(results, "adapter_p1_logprob0", abs(segs[0].avg_logprob) < 1e-9)
    _check(results, "adapter_p1_nospeech0", abs(segs[0].no_speech_prob) < 1e-9)
    _check(results, "adapter_p05_logprob", abs(segs[1].avg_logprob - math.log(0.5)) < 1e-6)
    _check(results, "adapter_p05_nospeech", abs(segs[1].no_speech_prob - 0.5) < 1e-9)
    _check(results, "adapter_nan_neutral",
           segs[2].no_speech_prob == 0.0 and segs[2].avg_logprob == 0.0)

    # -- WhisperCppBackend via injected fake model: shape + kwarg mapping --
    fake = FakeWCModel([FakeSeg(" copy that", 0.9)])
    be = core.WhisperCppBackend("large-v3", {"language": "en"}, model=fake)
    out_segs, info = be.transcribe(
        np.zeros(1600, dtype=np.float32),
        language="en", beam_size=5, vad_filter=True,
        condition_on_previous_text=False, initial_prompt="dispatch",
        no_speech_threshold=0.6, log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4, no_repeat_ngram_size=3,
        temperature=[0.0, 0.2])
    _check(results, "backend_returns_tuple",
           isinstance(out_segs, list) and isinstance(info, dict))
    _check(results, "backend_seg_shape",
           out_segs[0].text == " copy that" and hasattr(out_segs[0], "no_speech_prob"))
    _check(results, "backend_info_backend", info.get("backend") == "whispercpp")
    _check(results, "backend_audio_float32", fake.last_audio.dtype == np.float32)
    kw = fake.last_kwargs
    _check(results, "backend_no_context", kw.get("no_context") is True)
    _check(results, "backend_extract_prob", kw.get("extract_probability") is True)
    _check(results, "backend_maps_no_speech", kw.get("no_speech_thold") == 0.6)
    _check(results, "backend_maps_logprob", kw.get("logprob_thold") == -1.0)
    _check(results, "backend_initial_prompt", kw.get("initial_prompt") == "dispatch")
    # faster-whisper-only kwargs must NOT leak into the pywhispercpp call.
    _check(results, "backend_drops_beam_size", "beam_size" not in kw)
    _check(results, "backend_drops_vad_filter", "vad_filter" not in kw)
    _check(results, "backend_drops_ngram", "no_repeat_ngram_size" not in kw)

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "backend test failed"
    print("BACKEND TEST: PASS")


if __name__ == "__main__":
    run()
