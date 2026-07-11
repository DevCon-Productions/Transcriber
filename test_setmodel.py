"""Unit test for Engine.set_model hot-swap logic (no GPU/network).

Stubs WhisperModel so we can verify: the transcriber's model reference is
swapped atomically, cfg is updated, the on_done callback fires, an unknown model
is handled gracefully, and selecting the current model is a no-op.
"""
import transcriber as core


class FakeModel:
    def __init__(self, name, **k):
        if name == "explode":
            raise RuntimeError("no such model")
        self.name = name


def run():
    core.WhisperModel = FakeModel
    results = {}

    # Build an engine without touching audio/file logging/auth. Force the ct2
    # backend so this exercises the WhisperModel hot-swap path regardless of the
    # host architecture (on native ARM64 the default would be whisper.cpp).
    cfg = {"model": "large-v3", "device": "cuda", "compute_type": "float16",
           "engine": "ct2", "vad": {}, "filters": {}}
    eng = core.Engine(cfg, console=False, file_logging=False, enable_audio=False)
    eng.load_model()  # uses FakeModel
    results["initial_model"] = (eng.model.name == "large-v3")

    # Successful swap.
    out = {}
    eng.set_model("distil-large-v3", on_done=lambda ok, msg: out.update(ok=ok, msg=msg))
    results["swap_ok_callback"] = (out.get("ok") is True)
    results["engine_model_swapped"] = (eng.model.name == "distil-large-v3")
    results["transcriber_model_swapped"] = (eng.transcriber.model.name == "distil-large-v3")
    results["cfg_updated"] = (eng.cfg["model"] == "distil-large-v3")

    # No-op when selecting the current model.
    out2 = {}
    eng.set_model("distil-large-v3", on_done=lambda ok, msg: out2.update(ok=ok, msg=msg))
    results["noop_reports_ok"] = (out2.get("ok") is True and "Already" in out2.get("msg", ""))

    # Failure path: bad model -> ok False, old model retained.
    out3 = {}
    eng.set_model("explode", on_done=lambda ok, msg: out3.update(ok=ok, msg=msg))
    results["fail_callback_false"] = (out3.get("ok") is False)
    results["model_retained_on_fail"] = (eng.model.name == "distil-large-v3")
    results["cfg_retained_on_fail"] = (eng.cfg["model"] == "distil-large-v3")

    eng.shutdown()

    print("RESULTS:")
    ok = True
    for k, v in results.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "set_model test failed"
    print("SETMODEL TEST: PASS")


if __name__ == "__main__":
    run()
