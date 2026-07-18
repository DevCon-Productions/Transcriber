"""Unit tests for address/street extraction (no GPU/network).

Uses real transcript lines captured from the scanner logs, including the tricky
false-positive 'X and Y' phrases (conscious and alert, salt and pepper, ...)."""
from transcriber import extract_addresses, maps_url


def run():
    r = {}
    def names(txt):
        return [a[0] for a in extract_addresses(txt)]

    # --- real POSITIVES (from logs) -----------------------------------------
    r["numbered_full"] = ("3658 East 149th Street" in names("heading to 3658 East 149th Street"))
    # must not swallow the trailing word after the street type.
    r["no_trailing_word"] = (names("suspect heading to 3658 East 149th Street now")
                             == ["3658 East 149th Street"])
    r["numbered_drive"] = ("66745 Schubert Drive" in names("66745 Schubert Drive."))
    r["numbered_blvd"] = ("162 America Boulevard" in names("162 America Boulevard."))
    r["named_blvd"] = ("American Boulevard" in names("echo call over at American Boulevard for a 7-1-0 mail"))
    r["intersection"] = ("Detroit and Dover" in names("Detroit and Dover."))
    r["intersection2"] = ("Revere and Butternut" in names("Engine two en route, Revere and Butternut."))
    r["intersection_dir"] = ("East 93rd and Union" in names("Adam 3-1, can I get you for this DWI on East 93rd and Union?"))
    r["intersection_name"] = ("Sheldon and Eastland" in names("Sheldon and Eastland, King, Boy, Queen, 1-9-4-9."))

    # --- real FALSE-POSITIVE traps (must find NOTHING) ----------------------
    r["not_conscious_alert"] = (names("This male is sitting over at Conscious and Alert.") == [])
    r["not_salt_pepper"] = (names("White male, salt and pepper hair, blue shirt") == [])
    r["not_signs_vehicles"] = (names("being on stop signs, you know, and vehicles, and.") == [])
    r["not_vans_sneakers"] = (names("Great Vans and Sneakers.") == [])
    r["not_alert_responsive"] = (names("Use alert and responsive.") == [])
    r["not_thank_you"] = (names("Thank you and I appreciate it. Thank you again.") == [])
    r["not_unit_num"] = (names("Unit 306, you guys be able to assist?") == [])
    r["not_plain_chatter"] = (names("Okay, copy that, thank you.") == [])
    r["empty_safe"] = (names("") == [])

    # --- map URL building ---------------------------------------------------
    u = maps_url("3658 East 149th", "Cleveland, OH")
    r["url_has_query"] = ("google.com/maps/search" in u and "3658" in u)
    r["url_has_city"] = ("Cleveland" in u and "OH" in u)
    r["url_encoded"] = (" " not in u)   # spaces must be percent-encoded
    r["url_no_city"] = ("google.com/maps" in maps_url("Main and 5th"))

    print("RESULTS:")
    ok = True
    for k, v in r.items():
        print(f"  {'ok ' if v else 'FAIL'} {k}")
        ok = ok and v
    assert ok, "address test failed"
    print("ADDRESS TEST: PASS")


if __name__ == "__main__":
    run()
