"""Unit tests for TTS routing + config (no real speech, no GPU).

Verifies which transcript lines get read aloud based on feed selection, keyword
matching, and mode -- the decision logic in Engine._maybe_speak."""
import transcriber as core


class FakeTTS:
    available = True
    _voice_id = "test"
    def __init__(self): self.said = []
    def start(self): pass
    def say(self, txt): self.said.append(txt)
    def set_muted(self, m): pass
    def close(self): pass


def make_engine(mode, feeds, keywords, enabled=True):
    eng = core.Engine.__new__(core.Engine)
    eng.out = core.Output(console=False, file_logging=False)
    eng.tts_enabled = enabled
    eng.tts_mode = mode
    eng.tts_feeds = set(feeds)
    eng.tts_keywords = [k.lower() for k in keywords]
    eng.tts = FakeTTS()
    eng._ensure_tts = lambda: True
    return eng


def run():
    r = {}

    # --- feeds mode: only selected feeds spoken -----------------------------
    e = make_engine("feeds", ["West"], [])
    e._maybe_speak("West", "Adam 33 en route")
    e._maybe_speak("East", "Copy that")
    r["feeds_only_selected"] = (e.tts.said == ["Adam 33 en route"])

    # --- keywords mode: only keyword matches, any feed ----------------------
    e = make_engine("keywords", ["West"], ["shots fired", "pursuit"])
    e._maybe_speak("West", "routine traffic stop")          # no keyword -> skip
    e._maybe_speak("East", "reports of shots fired downtown")  # keyword -> speak
    e._maybe_speak("East", "in PURSUIT northbound")          # case-insensitive
    r["keywords_only_matches"] = (e.tts.said ==
                                  ["reports of shots fired downtown", "in PURSUIT northbound"])

    # --- both mode: selected feed OR keyword --------------------------------
    e = make_engine("both", ["West"], ["fire"])
    e._maybe_speak("West", "anything from west")   # feed match
    e._maybe_speak("East", "structure fire on 5th")  # keyword match
    e._maybe_speak("East", "nothing special")        # neither -> skip
    r["both_feed_or_kw"] = (e.tts.said ==
                            ["anything from west", "structure fire on 5th"])

    # --- disabled: nothing spoken -------------------------------------------
    e = make_engine("feeds", ["West"], [], enabled=False)
    e._maybe_speak("West", "should not speak")
    r["disabled_silent"] = (e.tts.said == [])

    # --- no keywords in keywords mode -> nothing ----------------------------
    e = make_engine("keywords", ["West"], [])
    e._maybe_speak("West", "anything")
    r["keywords_empty_silent"] = (e.tts.said == [])

    # --- word-boundary keyword matching (no substring false positives) ------
    km = core.keyword_matches
    # The exact reported bug: 'od' must NOT match inside 'understood'.
    r["no_substring_od"] = (km("Understood. Thank you.", ["od"]) is False)
    r["no_substring_gun"] = (km("the shift has begun", ["gun"]) is False)
    r["no_substring_fire"] = (km("by the campfire", ["fire"]) is False)
    # Real matches still work.
    r["match_word_od"] = (km("possible OD here", ["od"]) is True)
    r["match_word_gun"] = (km("he has a gun", ["gun"]) is True)
    r["match_phrase"] = (km("reports of shots fired", ["shots fired"]) is True)
    r["match_phrase_spaces"] = (km("shots   fired", ["shots fired"]) is True)
    r["match_hyphen"] = (km("a break-in occurred", ["break-in"]) is True)
    r["match_case_insensitive"] = (km("STRUCTURE FIRE", ["fire"]) is True)
    r["empty_safe"] = (km("", ["fire"]) is False and km("anything", []) is False)

    # Routing uses word-boundary matching end-to-end.
    e = make_engine("keywords", [], ["od"])
    e._maybe_speak("West", "Understood, en route")   # must NOT speak
    e._maybe_speak("West", "possible OD reported")     # must speak
    r["routing_word_boundary"] = (e.tts.said == ["possible OD reported"])

    # --- helpers ------------------------------------------------------------
    r["tts_available_bool"] = isinstance(core.tts_available(), bool)
    r["list_voices_list"] = isinstance(core.list_tts_voices(), list)

    print("RESULTS:")
    ok = True
    for k, v in r.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "tts test failed"
    print("TTS TEST: PASS")


if __name__ == "__main__":
    run()
