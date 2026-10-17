"""Build the ISR-flavoured evaluation set (eval/isr/) from hand-written reports.

    python scripts/build_isr_set.py

All reports are fictional and written for this project (MIT licence like the rest of the
repository): places, vessels, units and callsigns are invented. Each topic comes as a PAIR
of near-identical reports that differ in the facts, so retrieval must pick the right one
and a model is tempted to borrow a fact from the sibling. Pairs are assigned whole to
dev or test, so no pair straddles the split.

Question kinds, encoded in the id prefix:
  A   answerable from one report
  M   answerable only by comparing the two reports of a pair
  U1  unanswerable: the fact is stated in no report
  U2  unanswerable: the fact is stated only in the SIBLING report (the bait case)
  U3  unanswerable: false premise (the question assumes something the report contradicts)

Distractor reports (templated filler on unrelated names) pad each corpus so that
retrieval has real work to do. A check at the end fails the build if an answerable
question's gold answer is missing from its report(s), or if a U1/U2/U3 question's
telltale words appear where they would make it answerable.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "isr"

# Each pair: (split, [(doc_id, text), (doc_id, text)], [questions]).
# Question: (kind, text, gold, alternates, gold_doc_ids)
PAIRS = [
    # ---------------------------------------------------------------- dev
    (
        "dev",
        [
            ("d01", "SPOTREP 0114/Mar. At 0412Z the coastal radar post at Cape Orlin reported a bulk carrier, MV Karelia Star, steaming south-southeast at 11 knots about 14 nautical miles off the headland. The vessel is Panamanian-flagged and sits low in the water, suggesting a full load. No AIS transmission was received between 0350Z and 0430Z, although the vessel had been broadcasting normally earlier in the watch. The harbour pilot station was not contacted by the ship. The post assesses the AIS gap as probably an equipment fault, low confidence. Observation was by radar and one visual check through the night scope; hull colour was dark blue with a white superstructure."),
            ("d02", "SPOTREP 0115/Mar. At 0436Z the same post at Cape Orlin tracked a general cargo ship, MV Karelia Sun, heading north-northwest at 8 knots about 9 nautical miles off the headland. The vessel is Liberian-flagged and rides high in the water, suggesting it is empty or lightly loaded. Its AIS was transmitting throughout. The ship called the harbour pilot station at 0441Z and asked for an anchorage berth at the outer roads. The post assesses routine traffic. Observation was by radar and AIS only; weather prevented a visual check. Hull colour could not be determined."),
        ],
        [
            ("A", "What flag did the bulk carrier seen off Cape Orlin fly?", "Panamanian", (), ("d01",)),
            ("A", "Between which times did the Karelia Star's AIS go quiet?", "0350Z", ("0350",), ("d01",)),
            ("A", "What did the Cape Orlin post think caused the AIS gap?", "equipment fault", (), ("d01",)),
            ("A", "Which ship asked the pilot station for an anchorage berth?", "Karelia Sun", (), ("d02",)),
            ("A", "How fast was the Karelia Sun travelling?", "8 knots", ("8 kn", "eight knots"), ("d02",)),
            ("A", "In what direction was the Liberian-flagged ship heading?", "north-northwest", (), ("d02",)),
            ("M", "Which of the two ships off Cape Orlin was moving faster?", "Karelia Star", (), ("d01", "d02")),
            ("U1", "How many crew were aboard the Karelia Star?", None, (), ()),
            ("U1", "Which port did the Karelia Sun last sail from?", None, (), ()),
            ("U2", "What was the hull colour of the Karelia Sun?", None, (), ()),
            ("U3", "Why did the Karelia Star contact the harbour pilot station?", None, (), ()),
        ],
    ),
    (
        "dev",
        [
            ("d03", "TRACKREP A-41. Air surveillance radar at Ridge Station held track 7731 from 1902Z to 1927Z, a twin-engine turboprop on a steady heading of 070 degrees at 18,000 feet and 240 knots groundspeed. The track squawked a civil transponder code, 4417, and filed no flight plan that the station could find. It passed 22 nautical miles south of the Halvard airstrip without descending. The operator rated the track low interest. Track was lost in ground clutter at 1927Z over the Dunmere hills; no hand-off to the neighbouring sector was made."),
            ("d04", "TRACKREP A-42. Ridge Station held track 7740 from 1931Z to 1958Z, a single-engine piston aircraft heading 250 degrees at 6,500 feet and 110 knots groundspeed. It squawked 2203 and matched a filed flight plan from Halvard airstrip to Tarn Field. The track descended steadily after 1950Z and was handed off to Tarn Field approach at 1958Z. The operator rated it routine. Radar coverage was good throughout, with no clutter loss."),
        ],
        [
            ("A", "What transponder code did track 7731 squawk?", "4417", (), ("d03",)),
            ("A", "At what altitude was the twin-engine turboprop flying?", "18,000 feet", ("18000",), ("d03",)),
            ("A", "Where was track 7731 lost?", "Dunmere hills", (), ("d03",)),
            ("A", "Where was the piston aircraft's filed flight plan taking it?", "Tarn Field", (), ("d04",)),
            ("A", "What groundspeed did track 7740 have?", "110 knots", ("110 kn",), ("d04",)),
            ("A", "To whom was track 7740 handed off?", "Tarn Field approach", (), ("d04",)),
            ("M", "Which of the two tracks flew higher?", "7731", ("turboprop",), ("d03", "d04")),
            ("U1", "What was the registration of the aircraft on track 7731?", None, (), ()),
            ("U2", "What was the destination of track 7731?", None, (), ()),
            ("U3", "At what time was track 7731 handed off to the neighbouring sector?", None, (), ()),
        ],
    ),
    (
        "dev",
        [
            ("d05", "PATROL SUMMARY 3-12. A road patrol at checkpoint Birch observed a convoy of six trucks and one fuel bowser moving east on Route 9 at 0640 local. Vehicles were painted in a sand pattern and carried no unit markings. The convoy kept roughly 80 metres between vehicles at about 35 km/h. A staff car joined at the rear near the Marl bridge. The patrol photographed the lead truck and the bowser. The convoy did not stop at the checkpoint and could not be challenged, as the patrol had no vehicle to follow. Assessment: probable military logistic movement, medium confidence."),
            ("d06", "PATROL SUMMARY 3-13. A road patrol at checkpoint Cedar observed a convoy of three buses moving west on Route 4 at 0715 local. The buses were white with a blue stripe and showed the operator name Lindqvist Tours on the side. Spacing was about 40 metres at around 50 km/h. The convoy stopped at the checkpoint, passengers were checked, and all documents were found in order. The patrol photographed the lead bus. Assessment: civilian charter movement, high confidence."),
        ],
        [
            ("A", "How many trucks were in the convoy at checkpoint Birch?", "six", ("6",), ("d05",)),
            ("A", "What pattern were the Birch convoy vehicles painted in?", "sand", (), ("d05",)),
            ("A", "Where did the staff car join the convoy?", "Marl bridge", (), ("d05",)),
            ("A", "Which route were the buses on?", "Route 4", (), ("d06",)),
            ("A", "What operator name was on the buses?", "Lindqvist Tours", (), ("d06",)),
            ("A", "How confident was the assessment of the bus convoy?", "high confidence", ("high",), ("d06",)),
            ("M", "Which of the two convoys was travelling faster?", "bus", ("Cedar", "Lindqvist"), ("d05", "d06")),
            ("U1", "What was the licence plate of the lead truck?", None, (), ()),
            ("U2", "Did the buses carry unit markings?", None, (), ()),
            ("U3", "Why did the convoy at checkpoint Birch stop?", None, (), ()),
        ],
    ),
    (
        "dev",
        [
            ("d07", "EMITTER NOTE 5-08. Direction-finding site Fenn recorded a pulsed emitter on 9.41 GHz at 2105Z, bearing 315 degrees from the site. Pulse width was 1.2 microseconds with a repetition interval of 1,000 microseconds. The signal was present for 14 minutes before it stopped. Signal strength was strong, suggesting range under 30 kilometres. The parameters match a commercial marine navigation radar. No second site obtained a bearing, so no position fix was possible."),
            ("d08", "EMITTER NOTE 5-09. Direction-finding site Gorse recorded a pulsed emitter on 3.05 GHz at 2230Z, bearing 040 degrees from the site. Pulse width was 0.8 microseconds with a repetition interval of 2,400 microseconds. The signal was present for 3 minutes. Signal strength was weak. The parameters do not match any entry in the local emitter library. Sites Gorse and Hale both obtained bearings, giving a position fix about 60 kilometres north-east of Gorse."),
        ],
        [
            ("A", "What frequency did the emitter heard at site Fenn use?", "9.41 GHz", (), ("d07",)),
            ("A", "How long was the Fenn signal present?", "14 minutes", (), ("d07",)),
            ("A", "Why was no position fix possible for the Fenn emitter?", "second site", (), ("d07",)),
            ("A", "What was the pulse repetition interval of the Gorse emitter?", "2,400", ("2400",), ("d08",)),
            ("A", "Which other site obtained a bearing on the Gorse emitter?", "Hale", (), ("d08",)),
            ("A", "On what bearing was the Gorse emitter heard?", "040", (), ("d08",)),
            ("M", "Which of the two emitters operated on the higher frequency?", "Fenn", ("9.41",), ("d07", "d08")),
            ("U1", "What was the modulation type of the Fenn emitter?", None, (), ()),
            ("U2", "What was the estimated position of the Fenn emitter?", None, (), ()),
            ("U3", "Which radar type did the Gorse emitter match?", None, (), ()),
        ],
    ),
    (
        "dev",
        [
            ("d09", "IMAGERY NOTE 6-21. Satellite pass of 14 March over Kellan airfield, resolution 0.5 metres, cloud cover 10 percent. Compared with the 2 March pass, two hangars were completed on the north apron and a new 300-metre taxiway links the apron to the main runway. Eleven aircraft were visible on the ramp, up from seven. Fuel tanks near the east gate were unchanged. No vehicle activity was seen on the runway. The analyst assesses an expansion of apron capacity, high confidence."),
            ("d10", "IMAGERY NOTE 6-22. Satellite pass of 14 March over Dorne airfield, resolution 0.5 metres, cloud cover 65 percent. Cloud obscured the southern half, including the runway threshold. Compared with the 2 March pass, the northern ramp showed five aircraft, unchanged, and a perimeter fence was extended by about 200 metres on the west side. One fuel tank near the control tower appeared to have been removed. The analyst assesses no significant change, low confidence because of cloud."),
        ],
        [
            ("A", "How many aircraft were visible on Kellan's ramp?", "Eleven", ("11",), ("d09",)),
            ("A", "How long is the new taxiway at Kellan?", "300", (), ("d09",)),
            ("A", "How many hangars were completed at Kellan?", "two", ("2",), ("d09",)),
            ("A", "How much cloud covered Dorne?", "65 percent", ("65%",), ("d10",)),
            ("A", "By how much was Dorne's perimeter fence extended?", "200 metres", ("200 m",), ("d10",)),
            ("A", "Why was the Dorne assessment made with low confidence?", "cloud", (), ("d10",)),
            ("M", "Which airfield showed more aircraft on its ramp?", "Kellan", (), ("d09", "d10")),
            ("U1", "What aircraft types were parked on Kellan's ramp?", None, (), ()),
            ("U2", "How many hangars were completed at Dorne?", None, (), ()),
            ("U3", "Which of Kellan's fuel tanks was removed?", None, (), ()),
        ],
    ),
    (
        "dev",
        [
            ("d11", "SENSOR LOG 7-03. Coastal radar Cape Orlin was offline from 0200Z to 0545Z on 9 March for scheduled maintenance of the rotary joint. Two technicians replaced a worn bearing and recalibrated azimuth, finishing with an alignment error of 0.2 degrees. Coverage during the outage was provided by the mobile radar at Pike Hill. The radar returned to service at 0545Z and passed the acceptance check on the first attempt."),
            ("d12", "SENSOR LOG 7-04. Coastal radar Torre Point was offline from 1130Z to 1410Z on 11 March after a lightning strike damaged the transmitter power supply. One technician replaced the supply unit. Azimuth alignment was not affected. No backup radar was available during the outage, leaving a coverage gap in the northern sector. The radar returned to service at 1410Z but failed the acceptance check once and passed on the second attempt."),
        ],
        [
            ("A", "What part was replaced during the Cape Orlin maintenance?", "bearing", (), ("d11",)),
            ("A", "What was the azimuth alignment error after recalibration at Cape Orlin?", "0.2 degrees", (), ("d11",)),
            ("A", "What covered the Cape Orlin sector during the outage?", "Pike Hill", (), ("d11",)),
            ("A", "What caused the Torre Point outage?", "lightning", (), ("d12",)),
            ("A", "How many technicians worked on the Torre Point repair?", "One", ("1",), ("d12",)),
            ("A", "How many attempts did Torre Point need to pass its acceptance check?", "second", ("two", "twice", "2"), ("d12",)),
            ("M", "Which of the two radars was offline longer?", "Cape Orlin", (), ("d11", "d12")),
            ("U1", "What was the cost of the repair at Torre Point?", None, (), ()),
            ("U2", "Which backup radar covered Torre Point during its outage?", None, (), ()),
            ("U3", "What did lightning damage at Cape Orlin?", None, (), ()),
        ],
    ),
    # --------------------------------------------------------------- test
    (
        "test",
        [
            ("d13", "SPOTREP 0231/Apr. At 2218Z patrol boat Wren, on station near Saltholm Sound, sighted a fishing trawler, MV Aldous Bay, drifting with engines stopped about 4 nautical miles west of Saltholm light. The trawler is Danish-flagged and showed no navigation lights. Three persons were seen on deck. The trawler did not answer VHF calls on channel 16 over ten minutes. Wren held position at 1 nautical mile and photographed the vessel. The crew assesses possible engine failure or a radio problem; low confidence."),
            ("d14", "SPOTREP 0232/Apr. At 2250Z patrol boat Wren sighted a tanker, MV Orsa Bright, steaming east at 13 knots about 7 nautical miles north of Saltholm light. The tanker is Norwegian-flagged and displayed full navigation lights. It answered a VHF call on channel 16 within one minute and gave its destination as Brevik. Eleven persons were seen on the bridge wings through the night scope. Wren photographed the vessel at 2 nautical miles. Assessment: routine transit, high confidence."),
        ],
        [
            ("A", "What flag does the trawler near Saltholm light fly?", "Danish", (), ("d13",)),
            ("A", "How many people were seen on the trawler's deck?", "Three", ("3",), ("d13",)),
            ("A", "For how long did the trawler fail to answer VHF calls?", "ten minutes", ("10 minutes",), ("d13",)),
            ("A", "Where was the tanker bound?", "Brevik", (), ("d14",)),
            ("A", "How fast was the tanker steaming?", "13 knots", ("13 kn",), ("d14",)),
            ("A", "Which VHF channel did Wren call on?", "channel 16", ("16",), ("d13",)),
            ("M", "Which of the two vessels was showing navigation lights?", "Orsa Bright", ("tanker",), ("d13", "d14")),
            ("U1", "Who owned the trawler?", None, (), ()),
            ("U2", "Where was the trawler bound?", None, (), ()),
            ("U3", "What speed was the trawler making when it turned east?", None, (), ()),
        ],
    ),
    (
        "test",
        [
            ("d15", "TRACKREP B-17. Radar Hollin held track 2208 from 0310Z to 0344Z, a four-engine jet transport at 33,000 feet heading 095 degrees at 470 knots groundspeed. It squawked 1126 and was on a filed flight plan from Kasra to Venn. The track left coverage to the east at 0344Z and was handed off to Venn centre. Operator rating: routine."),
            ("d16", "TRACKREP B-18. Radar Hollin held track 2215 from 0322Z to 0331Z, a small helicopter at 900 feet heading 180 degrees at 80 knots groundspeed. It squawked no code and filed no flight plan. It hovered for two minutes near the Brack quarry at 0327Z before continuing south. The track was lost at 0331Z in low-level clutter. Operator rating: unknown, request follow-up."),
        ],
        [
            ("A", "What transponder code did track 2208 squawk?", "1126", (), ("d15",)),
            ("A", "At what altitude was the jet transport flying?", "33,000 feet", ("33000",), ("d15",)),
            ("A", "Where was the flight plan of track 2208 taking it?", "Venn", (), ("d15",)),
            ("A", "How long did the helicopter hover?", "two minutes", ("2 minutes",), ("d16",)),
            ("A", "What was the helicopter hovering near?", "Brack quarry", (), ("d16",)),
            ("A", "Why was track 2215 lost?", "clutter", (), ("d16",)),
            ("M", "Which of the two tracks was slower?", "2215", ("helicopter",), ("d15", "d16")),
            ("U1", "Who operated the helicopter?", None, (), ()),
            ("U2", "What was the destination of track 2215?", None, (), ()),
            ("U3", "What code did the helicopter squawk while hovering?", None, (), ()),
        ],
    ),
    (
        "test",
        [
            ("d17", "PATROL SUMMARY 8-02. A foot patrol at observation post Ash saw four tracked vehicles and two wheeled armoured vehicles parked in a tree line 1.5 kilometres north of Elm village at 0530 local. Engines were off. Camouflage netting covered three of the vehicles. About twenty personnel were seen cooking and checking equipment. The patrol sketched the layout. Assessment: probable company-size halt, medium confidence. The patrol withdrew at 0610 after a dog alerted near the position."),
            ("d18", "PATROL SUMMARY 8-03. A foot patrol at observation post Birchfield saw a farm tractor pulling a trailer of hay bales along the track south of Moss village at 0600 local. Two civilians walked alongside. No military equipment or markings were seen. The patrol watched for 25 minutes and took no photographs. Assessment: agricultural activity, high confidence. The patrol remained in position."),
        ],
        [
            ("A", "How many tracked vehicles were in the tree line near Elm?", "four", ("4",), ("d17",)),
            ("A", "What covered three of the vehicles at Elm?", "camouflage netting", (), ("d17",)),
            ("A", "Why did the Ash patrol withdraw?", "dog", (), ("d17",)),
            ("A", "What was the tractor pulling?", "trailer", ("hay",), ("d18",)),
            ("A", "How long did the Birchfield patrol watch the tractor?", "25 minutes", (), ("d18",)),
            ("A", "Where was the tractor seen?", "south of Moss village", ("Moss",), ("d18",)),
            ("M", "Which of the two patrols made the assessment with higher confidence?", "Birchfield", ("high confidence",), ("d17", "d18")),
            ("U1", "What was the name of the commander of the vehicles near Elm?", None, (), ()),
            ("U2", "At what time did the Birchfield patrol withdraw?", None, (), ()),
            ("U3", "How many personnel did the Birchfield patrol see cooking?", None, (), ()),
        ],
    ),
    (
        "test",
        [
            ("d19", "EMITTER NOTE 9-11. Site Larch recorded a continuous-wave carrier on 156.80 MHz at 1840Z, bearing 200 degrees. The carrier was frequency-modulated with voice and lasted 6 minutes. Signal strength was moderate. The frequency is the marine distress and calling channel, and the voice was in an unidentified language. Site Larch alone had coverage, so no fix was obtained. The recording was passed to the language team."),
            ("d20", "EMITTER NOTE 9-12. Site Moor recorded a frequency-hopping signal in the 225 to 400 MHz band at 1915Z, bearing 090 degrees. It hopped about 100 times per second and lasted 22 minutes. Signal strength was strong. The hopping pattern matched a known military radio family in the library. Sites Moor and Nye both obtained bearings, giving a fix about 25 kilometres east of Moor. A report was sent to the duty officer."),
        ],
        [
            ("A", "What frequency was the Larch carrier on?", "156.80", (), ("d19",)),
            ("A", "How long did the Larch transmission last?", "6 minutes", ("six minutes",), ("d19",)),
            ("A", "Who was the Larch recording passed to?", "language team", (), ("d19",)),
            ("A", "How many times per second did the Moor signal hop?", "100", (), ("d20",)),
            ("A", "How long did the Moor signal last?", "22 minutes", (), ("d20",)),
            ("A", "Where was the Moor emitter fixed?", "25 kilometres east of Moor", ("east of Moor",), ("d20",)),
            ("M", "Which of the two signals lasted longer?", "Moor", ("frequency-hopping",), ("d19", "d20")),
            ("U1", "What was the speaker's name on the Larch signal?", None, (), ()),
            ("U2", "What military radio family did the Larch signal match?", None, (), ()),
            ("U3", "How did site Larch obtain its position fix?", None, (), ()),
        ],
    ),
    (
        "test",
        [
            ("d21", "IMAGERY NOTE 10-04. Satellite pass of 20 April over Port Velis, resolution 0.3 metres, cloud cover 5 percent. Compared with the 6 April pass, three cranes were operating at the container quay, up from one. Fourteen ships were alongside or at anchor, up from nine. A new storage yard of roughly 120 containers had appeared south of the quay. The analyst assesses increased port throughput, medium confidence."),
            ("d22", "IMAGERY NOTE 10-05. Satellite pass of 20 April over Port Harrow, resolution 0.3 metres, cloud cover 40 percent. Compared with the 6 April pass, one crane was operating at the main quay, unchanged. Six ships were alongside or at anchor, down from eight. A dredger was working in the approach channel. The analyst assesses no significant change in throughput, medium confidence; cloud limited the view of the western berths."),
        ],
        [
            ("A", "How many cranes were operating at Port Velis?", "three", ("3",), ("d21",)),
            ("A", "How many ships were at Port Velis?", "Fourteen", ("14",), ("d21",)),
            ("A", "How large was the new storage yard at Port Velis?", "120 containers", ("120",), ("d21",)),
            ("A", "What was working in Port Harrow's approach channel?", "dredger", (), ("d22",)),
            ("A", "How many ships had been at Port Harrow on the earlier pass?", "eight", ("8",), ("d22",)),
            ("A", "What limited the analyst's view at Port Harrow?", "cloud", (), ("d22",)),
            ("M", "Which of the two ports had more ships?", "Velis", (), ("d21", "d22")),
            ("U1", "Which flags did the ships at Port Velis fly?", None, (), ()),
            ("U2", "How many containers were in the storage yard at Port Harrow?", None, (), ()),
            ("U3", "Which crane was removed from the quay at Port Velis?", None, (), ()),
        ],
    ),
    (
        "test",
        [
            ("d23", "SENSOR LOG 11-07. Seismic sensor array Quarry-North was offline from 0900Z to 1300Z on 3 May for scheduled firmware upgrade. Version 4.2 replaced version 3.9. One technician performed the upgrade remotely. Detection threshold was lowered from 12 to 10 units. Neighbouring array Quarry-East provided partial coverage during the outage. The array passed its self-test at 1300Z."),
            ("d24", "SENSOR LOG 11-08. Acoustic sensor Harbour-3 was offline from 2200Z on 5 May to 0630Z on 6 May after a fault in its hydrophone cable. Two divers replaced the cable section. The sensor's detection range is unchanged at about 5 kilometres. No other sensor covered the harbour mouth during the outage. The sensor passed its self-test at 0630Z."),
        ],
        [
            ("A", "Which firmware version replaced version 3.9?", "4.2", (), ("d23",)),
            ("A", "How was the Quarry-North upgrade carried out?", "remotely", (), ("d23",)),
            ("A", "What covered the Quarry-North area during the outage?", "Quarry-East", (), ("d23",)),
            ("A", "What caused the Harbour-3 outage?", "hydrophone cable", (), ("d24",)),
            ("A", "How many divers replaced the cable section?", "Two", ("2",), ("d24",)),
            ("A", "What is the detection range of Harbour-3?", "5 kilometres", ("5 km",), ("d24",)),
            ("M", "Which of the two sensors was offline longer?", "Harbour-3", (), ("d23", "d24")),
            ("U1", "What was the cost of the Quarry-North upgrade?", None, (), ()),
            ("U2", "What firmware version did Harbour-3 run?", None, (), ()),
            ("U3", "Which neighbouring sensor covered Harbour-3 during its outage?", None, (), ()),
        ],
    ),
]

# Words that must NOT appear in the pair's reports for a U1 question about them to be unanswerable.
FORBIDDEN_FOR_U1 = ["crew", "cargo", "registration", "modulation", "licence", "cost", "owned", "owner", "commander", "speaker", "operated the helicopter", "aircraft types", "flags did"]

PLACES = ["Ostrava Ridge", "Pellam Head", "Quint Basin", "Rasmus Flat", "Sable Cove", "Tamsin Pass", "Ulm Heath", "Varga Neck", "Wick Marsh", "Yarrow Gap", "Zeno Bluff", "Abbot Fen", "Brisk Tor", "Corvin Strait", "Drum Hollow", "Eskil Point"]
VESSELS = ["MV Lantern Tide", "MV Grey Heron", "MV Saffron Dawn", "MV Ironbark", "MV Pale Meridian", "MV Coral Anvil", "MV Westerly Fox", "MV Dunlin"]
FLAGS = ["Maltese", "Cypriot", "Bahamian", "Greek", "Finnish", "Estonian"]
HEADINGS = ["north", "south", "east", "west", "north-east", "south-west"]
ASSESS = ["routine traffic", "no action required", "of low interest", "within normal pattern"]


def distractors(split: str, n: int, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for i in range(n):
        kind = i % 4
        place, place2 = rng.sample(PLACES, 2)
        hh, mm = rng.randrange(24), rng.randrange(60)
        if kind == 0:
            text = (f"SPOTREP F-{split[0]}{i:02d}. At {hh:02d}{mm:02d}Z the post at {place} logged {rng.choice(VESSELS)}, {rng.choice(FLAGS)}-flagged, "
                    f"heading {rng.choice(HEADINGS)} at {rng.randrange(6, 16)} knots about {rng.randrange(3, 20)} nautical miles offshore. Assessment: {rng.choice(ASSESS)}.")
        elif kind == 1:
            text = (f"TRACKREP F-{split[0]}{i:02d}. Radar at {place} held a track from {hh:02d}{mm:02d}Z for {rng.randrange(5, 40)} minutes at {rng.randrange(2, 35) * 1000} feet "
                    f"heading {rng.randrange(0, 360):03d} degrees, {rng.randrange(90, 480)} knots groundspeed. Transponder code {rng.randrange(1000, 7777)}. Rated {rng.choice(ASSESS)}.")
        elif kind == 2:
            text = (f"PATROL SUMMARY F-{split[0]}{i:02d}. A patrol near {place} reported {rng.randrange(2, 9)} vehicles moving {rng.choice(HEADINGS)} on a minor road at {hh:02d}{mm:02d} local, "
                    f"spaced about {rng.randrange(30, 120)} metres apart. Nothing further observed toward {place2}. Assessment: {rng.choice(ASSESS)}.")
        else:
            text = (f"SENSOR LOG F-{split[0]}{i:02d}. The sensor at {place} was offline for {rng.randrange(10, 180)} minutes from {hh:02d}{mm:02d}Z for a routine check and returned to service. "
                    f"Coverage of {place2} was unaffected. Status: {rng.choice(ASSESS)}.")
        out[f"f{split[0]}{i:02d}"] = text
    return out


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        docs: dict[str, str] = {}
        items = []
        counts: dict[str, int] = {}
        for sp, pair_docs, qs in PAIRS:
            if sp != split:
                continue
            pair_text = " ".join(t for _, t in pair_docs).lower()
            for d, t in pair_docs:
                docs[d] = t
            for kind, question, gold, alts, gold_docs in qs:
                counts[kind] = counts.get(kind, 0) + 1
                qid = f"{kind}-{split}-{counts[kind]:02d}"
                answerable = kind in ("A", "M")
                if answerable:
                    assert gold and gold_docs, qid
                    in_docs = " ".join(norm(docs[d]) for d in gold_docs)
                    assert any(norm(g) in in_docs for g in (gold, *alts)), f"{qid}: gold {gold!r} not in {gold_docs}"
                if kind == "U1":
                    for w in FORBIDDEN_FOR_U1:
                        if w in norm(question) and w in pair_text:
                            raise AssertionError(f"{qid}: '{w}' appears in the pair's reports, so the question may be answerable")
                items.append({
                    "id": qid, "question": question, "answerable": answerable,
                    "gold_docs": list(gold_docs), "gold_answer": gold, "alt_answers": list(alts),
                })
        docs.update(distractors(split, n=40, seed=7 if split == "dev" else 8))
        (OUT / f"{split}_corpus.jsonl").write_text("".join(json.dumps({"doc_id": k, "text": v}) + "\n" for k, v in docs.items()), encoding="utf-8")
        (OUT / f"{split}_questions.jsonl").write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in items), encoding="utf-8")
        print(f"{split}: {len(docs)} documents ({len(docs) - 2 * sum(1 for p in PAIRS if p[0] == split)} distractors); questions {dict(sorted(counts.items()))}")

    lock = []
    for name in ("test_corpus.jsonl", "test_questions.jsonl"):
        lock.append(f"{hashlib.sha256((OUT / name).read_bytes()).hexdigest()}  {name}")
    (OUT / "LOCK.txt").write_text(
        "SHA-256 of the test files when first built. The test split is for one final run per configuration;\n"
        "no prompt, retrieval or threshold choice may be made on it. A change to these hashes means the set was edited.\n"
        + "\n".join(lock) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
