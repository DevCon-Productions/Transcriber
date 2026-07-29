"""Unit tests for extract_callsign -- precision-focused, using REAL transcript
lines captured from the Cleveland feeds plus crafted false-positive traps."""
import transcriber as core
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

    # --- aviation call signs, FR24 links, service profiles ------------------
    r = {}
    av = lambda t: core.extract_callsign(t, style="aviation")
    r["av_airline"] = (av("Kennedy Tower, Delta 510, ILS 22 left.") == "DELTA 510")
    r["av_united"] = (av("United 1685, contact departure.") == "UNITED 1685")
    r["av_speedbird"] = (av("Speedbird 117 heavy, climb and maintain.")
                         == "SPEEDBIRD 117")
    r["av_tail"] = (av("November 65 Juliet Charlie, taxi via Alpha.") == "N65JC")
    # Whisper hyphenates spoken digits unpredictably, and the label is the
    # grouping key for colour and click-to-filter -- both spellings must collapse
    # to one, or an aircraft scatters across several "units".
    r["av_hyphens_normalise"] = (av("Delta 5-10, wind check.") == "DELTA 510")
    r["av_same_label"] = (av("Delta 5-10 heavy") == av("Delta 510"))
    r["av_drops_heavy"] = (av("Speedbird 117 heavy") == av("Speedbird 117"))
    # Whisper splits compound telephony names and punctuates flight numbers
    # freely -- all of these are from real KCLE Tower transcripts.
    r["av_split_name"] = (av("South West 46, 94, contact departure.")
                          == "SOUTHWEST 4694")
    r["av_split_same_as_joined"] = (av("South West 46, 94")
                                    == av("Southwest 4694"))
    r["av_split_jetblue"] = (av("Jet Blue 231 heavy, turn right.") == "JETBLUE 231")
    # ...and a name that really contains a space may arrive joined.
    r["av_joined_air_france"] = (av("AirFrance 22, contact ground.")
                                 == av("Air France 22, contact ground."))
    # The comma rule must not invent flight numbers out of following digits,
    # nor throw away a call sign when it over-runs.
    r["av_comma_lone_digit"] = (av("Delta 510, 2 miles out.") == "DELTA 510")
    r["av_comma_overflow"] = (av("Delta 510, 20 miles out.") == "DELTA 510")
    r["av_comma_joins_pair"] = (av("United 16, 85 on frequency.") == "UNITED 1685")
    # Registrations must be shaped like real ones, or garbled phonetics become
    # dead FlightRadar24 links. This exact line came off KCLE Tower and produced
    # "N0" -> /aircraft/n0: "Fox Try" isn't Foxtrot, so nothing more decoded.
    r["av_tail_rejects_single"] = (
        av("November 0, Fox Try Yankee Cross, runway 86, sighted.") is None)
    r["av_tail_rejects_bare"] = (av("November, go ahead.") is None)
    r["av_tail_rejects_leading_zero"] = (av("November 0-1-2 Alpha.") is None)
    r["av_tail_still_works"] = (
        av("November 1-2-3 Alpha Bravo, cleared for takeoff.") == "N123AB")
    r["av_tail_letters"] = (av("November 774 Sierra Papa on the ground.")
                            == "N774SP")
    r["av_unknown_airline"] = (av("Skyhawk 12345, cleared to land.") is None)
    r["av_no_aircraft"] = (av("Cleared to land, runway 22 left.") is None)
    r["av_ignores_police"] = (av("Adam 33 responding to the call.") is None)
    # Styles don't bleed into each other.
    r["style_emergency_default"] = (cs("Engine 14 on scene.") == "ENGINE 14")
    r["style_none"] = (core.extract_callsign("Engine 14 on scene.",
                                             style=None) is None)

    ac = core.extract_aircraft("Delta 510 and November 65 Juliet Charlie.")
    r["air_two_found"] = (len(ac) == 2)
    r["air_idents"] = ([i for _s, i, _l in ac] == ["DL510", "N65JC"])
    r["air_flight_url"] = (core.aircraft_url("DL510")
                           == "https://www.flightradar24.com/data/flights/dl510")
    r["air_reg_url"] = (core.aircraft_url("N65JC")
                        == "https://www.flightradar24.com/data/aircraft/n65jc")
    r["air_empty_url"] = (core.aircraft_url("") == "")

    cfg = {"initial_prompt": "GLOBAL"}
    d = core.service_profile({}, cfg)          # an existing, untyped feed
    r["svc_default_prompt"] = (d["prompt"] == "GLOBAL")
    r["svc_default_callsigns"] = (d["callsigns"] == "emergency")
    r["svc_default_addresses"] = (d["address_links"] is True)
    atc = core.service_profile({"service": "atc"}, cfg)
    r["svc_atc_prompt"] = ("air traffic control" in atc["prompt"])
    r["svc_atc_callsigns"] = (atc["callsigns"] == "aviation")
    r["svc_atc_no_addresses"] = (atc["address_links"] is False)
    r["svc_atc_aircraft"] = (atc["aircraft_links"] is True)
    pol = core.service_profile({"service": "police"}, cfg)
    fire = core.service_profile({"service": "fire"}, cfg)
    r["svc_police_fire_differ"] = (pol["prompt"] != fire["prompt"])
    r["svc_police_fire_same_rest"] = (
        pol["callsigns"] == fire["callsigns"] == "emergency"
        and pol["address_links"] and fire["address_links"])
    r["svc_general_quiet"] = (core.service_profile({"service": "general"},
                                                   cfg)["callsigns"] is None)
    o = core.service_profile({"service": "atc", "initial_prompt": "MINE"}, cfg)
    r["svc_override_wins"] = (o["prompt"] == "MINE"
                              and o["callsigns"] == "aviation")
    r["svc_unknown_service"] = (
        core.service_profile({"service": "nonsense"}, cfg)["prompt"] == "GLOBAL")

    for k, v in r.items():
        print(f"{'ok ' if v else 'FAIL'} {k}")
        fails += not v

    assert fails == 0, f"{fails} case(s) failed"
    print("CALLSIGN TEST: PASS")

if __name__ == "__main__":
    run()
