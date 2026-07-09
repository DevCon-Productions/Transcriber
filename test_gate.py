"""Deterministic unit test for SpeechGate segmentation (no GPU/network)."""
import numpy as np
from transcriber import SpeechGate, FRAME_SAMPLES, FRAME_MS

def frames_from(signal):
    n = (len(signal) // FRAME_SAMPLES) * FRAME_SAMPLES
    return signal[:n].reshape(-1, FRAME_SAMPLES)

def make(seconds, amp, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * 16000)
    return (rng.standard_normal(n) * amp).astype(np.float32)

cfg = {"trigger_ratio": 3.0, "abs_min_rms": 0.004, "silence_hangover_sec": 0.5,
       "min_speech_sec": 0.4, "max_segment_sec": 25.0, "preroll_sec": 0.3}

# Build: 2s quiet noise floor, 1.5s loud "transmission", 1s silence, 1.2s loud, 1s silence
sig = np.concatenate([
    make(2.0, 0.003, 1),    # background
    make(1.5, 0.08, 2),     # transmission A
    make(1.0, 0.003, 3),    # gap
    make(1.2, 0.08, 4),     # transmission B
    make(1.0, 0.003, 5),    # trailing gap (flush B)
])

gate = SpeechGate(cfg)
segments = []
for fr in frames_from(sig):
    seg = gate.push(fr.copy())
    if seg is not None:
        segments.append(seg)

durs = [round(len(s)/16000, 2) for s in segments]
print(f"segments detected: {len(segments)}  durations(s): {durs}")
assert len(segments) == 2, f"expected 2 transmissions, got {len(segments)}"
# Each should be roughly the transmission length (+ preroll), well under input total.
assert all(1.0 < d < 3.0 for d in durs), f"unexpected durations: {durs}"

# --- Continuous mode (TV/app audio): emit fixed chunks, NO energy gating ----
# Quiet continuous audio that the legacy gate would drop must still be captured.
quiet = make(6.0, 0.001, 9)        # very quiet, well below abs_min_rms
cg = SpeechGate({"continuous": True, "chunk_sec": 1.0})
cont_segs = [s for fr in frames_from(quiet) if (s := cg.push(fr.copy())) is not None]
assert len(cont_segs) >= 5, f"continuous should chunk quiet audio, got {len(cont_segs)}"
# The same quiet audio is fully gated out by the legacy energy gate (the bug).
lg = SpeechGate(cfg)
legacy_segs = [s for fr in frames_from(quiet) if (s := lg.push(fr.copy())) is not None]
assert len(legacy_segs) == 0, f"legacy gate should drop quiet audio, got {len(legacy_segs)}"
print(f"continuous chunks: {len(cont_segs)}  legacy chunks: {len(legacy_segs)}")
print("GATE TEST: PASS")
