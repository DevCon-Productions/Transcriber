"""Unit tests for extract_callsign -- precision-focused, using REAL transcript
lines captured from the Cleveland feeds plus crafted false-positive traps."""
from transcriber import extract_callsign as cs

# (text, expected)  -- expected None means "must NOT detect a unit"
CASES = [
    # --- real positives (units self-identifying) ------------------------------
    ("Adam 33 for code 2.", "ADAM 33"),
    ("Adam 3-1, can I get you for this DWI on East 93rd and Union?", "ADAM 31"),
    ("Radio to Barney 36, can I get you for this code 2 in your zone?", "BARNEY 36"),
    ("Engine 14, show me en route to that call.", "ENGINE 14"),
    ("This is 31 on C. This is a man down.", None),   # bare number, no prefix
    ("Medic 7 en route.", "MEDIC 7"),
    ("Battalion 1 on scene.", "BATTALION 1"),

    # --- real NEGATIVES that must NOT be mistaken for units -------------------
    # License plate spelled phonetically (multiple phonetic words in a row):
    ("King Tom George, 9-0-5-1.", None),
    ("Illinois plate, Frank Paul, 872-682.", None),
    # Street addresses:
    ("heading to 3658 East 149th Street. That's 3658 East 149th.", None),
    ("He's at 8585, that's 8585, that's code 2.", None),   # bare numbers
    # Pure chatter, no unit:
    ("Okay, copy, thank you. You're welcome.", None),
    ("Can you run a mail and check priors?", None),
    ("", None),
    (None, None),
]

def run():
    fails = 0
    for text, expected in CASES:
        got = cs(text)
        ok = got == expected
        fails += not ok
        print(f"{'ok ' if ok else 'FAIL'} {got!r:>12}  <- {text!r}")
        if not ok:
            print(f"      expected {expected!r}")

    # Extensible local prefixes.
    got = cs("Zone 5 to dispatch", extra_prefixes=["zone"])
    print(f"{'ok ' if got=='ZONE 5' else 'FAIL'} extra_prefixes -> {got!r}")
    fails += got != "ZONE 5"

    assert fails == 0, f"{fails} case(s) failed"
    print("CALLSIGN TEST: PASS")

if __name__ == "__main__":
    run()
