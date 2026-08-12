"""Build the retrieval corpus and its labelled query set. RUN BY HAND.

    python tools/build_corpus.py            # writes data/corpus/
    python tools/build_corpus.py --stats    # regenerate and report, write nothing

PLAN.md §10a decided an authored synthetic corpus over a borrowed one, for two
reasons: the relevance judgments are the expensive part and are far cheaper
authored *with* the text than retrofitted onto it, and a corpus the model has
memorised makes groundedness and abstention unmeasurable in Capstone I.

## "Authored" means the generator, not the prose

Six thousand chunks of four hundred tokens is roughly 2.4 million tokens. What
is authored here is the template set, the entity model and the phenomena — not
2.4 million tokens of hand-written English. That is a real limitation and it is
stated rather than implied: template-generated text has genuine lexical
variety and genuine structure, and it is not natural prose. Retrieval numbers
measured on it describe this corpus.

What the generation buys in return is control. Every phenomenon a retrieval
lesson needs to demonstrate is planted deliberately, at a known rate, with the
gold chunk recorded at the moment the query is written — and
`tools/verify_corpus.py` then *measures* that each one is present rather than
trusting that it is.

## The world

A fictional logistics company, continuous with the Module 0 fixtures: shipment
`TL-4471`, the north-east depot, address-verification holds. Seven document
types in four registers, because vocabulary mismatch between registers is one
of the phenomena and cannot exist in a corpus written in one voice.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "corpus"

SEED = 20260811
TARGET_DOCS = 1200
N_QUERIES = 200

#: Documents rather than chunks, deliberately. Module 3 teaches chunking as a
#: decision with consequences, and a pre-chunked corpus removes the decision.
#: Gold labels are therefore at document granularity, which is also what IR
#: evaluation normally uses. `tools/record_embeddings.py` ships embeddings for
#: several fixed chunkings so the chunking lesson can compare them with real
#: vectors and no model in the browser.

DEPOTS = ["north-east", "south-west", "midlands", "north-west", "south-east", "scotland"]
CARRIERS = ["Halloway Freight", "Kestrel Logistics", "Ardent Carriage",
            "Bramble Transport", "Northgate Haulage"]

#: Register-specific vocabulary. The same concept, named differently by
#: customer-facing and operational documents — this is what makes vocabulary
#: mismatch a real phenomenon rather than an asserted one.
TERMS = {
    "parcel": {"faq": "parcel", "policy": "consignment", "ops": "unit",
               "legal": "the goods"},
    "delay": {"faq": "hold-up", "policy": "exception event", "ops": "dwell",
              "legal": "period of non-performance"},
    "refund": {"faq": "money back", "policy": "reimbursement", "ops": "credit note",
               "legal": "restitution"},
    "address_check": {"faq": "address check", "policy": "address verification hold",
                      "ops": "AV flag", "legal": "delivery-point confirmation"},
    "late": {"faq": "late", "policy": "outside the service window",
             "ops": "SLA breach", "legal": "in default of the delivery undertaking"},
}

#: Terminology drift over time: older documents use the first form.
DRIFT = {"consignment": "shipment", "depot": "hub", "manifest": "load sheet"}

POLICY_TOPICS = [
    ("hold_release", "Hold and release"),
    ("address_verification", "Address verification"),
    ("refund_eligibility", "Refund eligibility"),
    ("damage_claims", "Damage claims"),
    ("redelivery", "Redelivery attempts"),
    ("dangerous_goods", "Restricted and dangerous goods"),
    ("proof_of_delivery", "Proof of delivery"),
    ("weekend_service", "Weekend and holiday service"),
    ("oversize", "Oversize and palletised freight"),
    ("collection_points", "Collection points"),
    ("insurance", "Declared value and insurance"),
    ("returns", "Returns and refusals"),
]


def _para(rng: random.Random, sentences: list[str]) -> str:
    return " ".join(sentences)


def _register_word(concept: str, register: str) -> str:
    return TERMS[concept][register]


def _drift(text: str, era: str) -> str:
    """Older documents use the earlier term; newer ones the later."""
    if era == "old":
        return text
    for old, new in DRIFT.items():
        text = text.replace(old, new)
    return text


#: Additional paragraphs, per document type. Real operational documents carry a
#: great deal of text that is *not* the answer to anything, and a corpus of
#: four-sentence stubs makes retrieval unrealistically easy: every document is
#: almost pure signal, so a query that matches anything matches the right thing.
#: These pools restore the ratio.
FILLER = {
    "policy": [
        "Depot managers are responsible for ensuring that staff are briefed on "
        "this policy at the start of each shift, and a record of that briefing is "
        "retained for audit purposes for twelve months.",
        "Nothing in this policy limits a customer statutory rights, and where a "
        "conflict arises between this document and the carriage agreement, the "
        "carriage agreement prevails on matters of liability only.",
        "Exceptions to this policy may be granted by a regional manager in "
        "writing, and any such exception is recorded against the consignment so "
        "that it is visible on subsequent scans.",
        "This document is reviewed annually. Proposed amendments are circulated "
        "to depot managers four weeks before they take effect, and comments are "
        "considered by the operations board at its next scheduled meeting.",
        "Terminology in this document follows the operational glossary. Where a "
        "customer-facing publication uses a different term for the same concept, "
        "the operational term governs for the purposes of this policy.",
        "Training material derived from this policy is maintained separately and "
        "may lag a revision by up to one reporting period, which is why the "
        "effective date above is authoritative rather than the training deck.",
    ],
    "faq": [
        "We know waiting for a delivery is frustrating, and we try to keep the "
        "tracking page up to date, though there can be a short gap between "
        "something happening and the page showing it.",
        "If you would rather speak to somebody, the customer service desk is open "
        "on weekdays and can see the same information you can, plus the notes the "
        "depot has added.",
        "You can change the delivery address before the parcel leaves the depot it "
        "is currently at. Once it is out for delivery we cannot redirect it, "
        "though you can arrange a redelivery afterwards.",
        "Tracking updates come from scans, so a parcel that has not been scanned "
        "for a while has not necessarily stopped moving; it may simply not have "
        "passed a scan point.",
        "If a parcel is returned to the sender, the sender will usually contact "
        "you about what happens next, since the arrangement is between you and "
        "them rather than with us.",
    ],
    "handbook": [
        "Scanners are docked at the end of each shift and any unit reporting a "
        "battery fault is taken out of service and logged with the equipment team "
        "rather than left on the charging rack.",
        "The sortation area is walked at the end of each shift and any unit left "
        "in a lane is scanned to the exceptions bay so that it appears on the "
        "following morning report rather than being discovered days later.",
        "Agency staff working their first shift are paired with a permanent member "
        "of the team for the whole of that shift, and are not permitted to operate "
        "the exceptions bay unsupervised.",
        "Pallet movements between bays are recorded even when the pallet does not "
        "leave the building, because the dwell report is built from scan events "
        "rather than from movements.",
        "Health and safety briefings take place at the start of each shift and "
        "cover the specific hazards present that day, including any temporary "
        "changes to the traffic route through the yard.",
        "Where a label is damaged and cannot be scanned, the unit is taken to the "
        "relabelling station rather than being keyed in manually, so that the scan "
        "history remains continuous.",
    ],
    "incident": [
        "The duty supervisor was notified at the time and confirmed that the unit "
        "had been staged correctly, so no further corrective action was raised "
        "against the depot for this event.",
        "This incident was included in the weekly exception review. No trend was "
        "identified linking it to other events at the same depot in the preceding "
        "fortnight.",
        "Customer contact was attempted twice before the consignment was released, "
        "and both attempts are recorded against the consignment in the contact log.",
        "The scan history shows the unit passing the inbound point normally, so the "
        "exception was raised by the address check rather than by any physical "
        "handling issue.",
    ],
    "shipment": [
        "Scan events for this consignment are recorded at inbound, sortation and "
        "outbound, and are retained for ninety days from the final movement before "
        "being archived to cold storage.",
        "Where a consignment is held, the reason code is written against the most "
        "recent scan rather than against the consignment as a whole, so a "
        "consignment held twice carries two reason codes.",
        "Service level is measured from the first inbound scan rather than from the "
        "point of collection, which is why the tracking page and the service report "
        "occasionally disagree by a day.",
    ],
    "agreement": [
        "The parties shall review this agreement annually, and either party may "
        "propose amendments in writing not less than sixty days before the "
        "anniversary of the commencement date.",
        "Neither party shall be liable for failure to perform to the extent that "
        "such failure arises from circumstances beyond its reasonable control, "
        "provided that the affected party notifies the other promptly.",
        "This clause shall survive termination of the agreement to the extent "
        "necessary to give effect to the rights and obligations accrued before the "
        "date of termination.",
        "Notices under this agreement shall be given in writing to the addresses "
        "specified in the schedule, and shall be deemed received on the second "
        "working day after posting.",
    ],
    "release": [
        "This release was deployed outside core operating hours and required no "
        "downtime. Depots reported no issues during the following shift.",
        "Reporting definitions are unchanged by this release, so figures before and "
        "after the deployment remain directly comparable.",
        "The change is behind a configuration flag and can be reverted without a "
        "further deployment if an issue is identified.",
    ],
}


#: Filler paragraphs drawn per document, as a (low, high) range for
#: `random.randrange`. Two properties have to hold at once, and getting one
#: without the other was the failure mode both earlier versions hit:
#:
#:   * spread *within* a type -- a fixed count made every handbook 397-398
#:     tokens, which is template regularity showing through and gives chunk-size
#:     choices nothing to bite on;
#:   * ordering *across* types -- a single wide range for everything made
#:     policies and shipment records the same average length, which no real
#:     operational corpus looks like.
#:
#: So the range is per type: policies run long, release notes run short, and
#: each spans roughly a factor of three from end to end.
FILLER_RANGE = {
    "policy":    (6, 22),
    "handbook":  (5, 20),
    "agreement": (4, 16),
    "faq":       (3, 12),
    "incident":  (3, 12),
    "shipment":  (2, 11),
    "release":   (2, 9),
}


def _expand(rng: random.Random, doc: dict) -> dict:
    """Pad a document with type-appropriate text that is not the answer."""
    pool = FILLER.get(doc["type"], [])
    if pool:
        # Drawn *with replacement*, so the count is free to exceed the pool
        # size. Repeated boilerplate is what operational documents genuinely
        # contain, and it is also what makes near-duplicate detection a real
        # problem rather than a toy one.
        low, high = FILLER_RANGE.get(doc["type"], (3, 8))
        doc["text"] = doc["text"] + " " + " ".join(rng.choices(pool, k=rng.randrange(low, high)))
    return doc


# --- document generators ----------------------------------------------------


def policy_doc(rng: random.Random, topic: tuple[str, str], version: int,
               effective: str, superseded_by: str | None) -> dict:
    key, title = topic
    hours = rng.choice([24, 48, 72])
    days = rng.choice([7, 14, 21, 28])
    body = [
        f"{title} policy, version {version}, effective {effective}.",
        f"This policy governs how a {_register_word('parcel', 'policy')} is "
        f"handled once an {_register_word('address_check', 'policy')} has been "
        f"raised against it.",
        f"An {_register_word('address_check', 'policy')} pauses movement for up "
        f"to {hours} hours while the delivery point is confirmed.",
        f"Where confirmation is not obtained within {hours} hours, the "
        f"consignment is returned to the originating depot and the customer is "
        f"notified.",
        f"A claim for {_register_word('refund', 'policy')} arising from an "
        f"{_register_word('delay', 'policy')} must be submitted within {days} "
        f"days of the original service date.",
        f"Claims submitted after {days} days are assessed at the discretion of "
        f"the depot manager and are not guaranteed.",
    ]
    if superseded_by:
        body.insert(1, f"This version is superseded by {superseded_by} and is "
                       f"retained for reference only.")
    era = "old" if version == 1 else "new"
    return {
        "doc_id": f"policy-{key}-v{version}",
        "type": "policy",
        "title": f"{title} policy v{version}",
        "effective": effective,
        "superseded": bool(superseded_by),
        "text": _drift(_para(rng, body), era),
        "facts": {"hold_hours": hours, "claim_days": days, "topic": key,
                  "version": version},
    }


def faq_entry(rng: random.Random, topic: tuple[str, str], hours: int, days: int,
              contradicts: bool) -> dict:
    key, title = topic
    # A deliberately wrong FAQ: the numbers disagree with the policy, and the
    # policy is authoritative. Retrieval that prefers the customer-facing
    # register gets the wrong answer confidently.
    stated_hours = hours + 24 if contradicts else hours
    stated_days = days - 7 if contradicts else days
    body = [
        f"How long can my {_register_word('parcel', 'faq')} be held?",
        f"If there is an {_register_word('address_check', 'faq')} on your "
        f"{_register_word('parcel', 'faq')}, we hold it for up to "
        f"{stated_hours} hours while we confirm where it is going.",
        f"If we cannot confirm the address in that time, the "
        f"{_register_word('parcel', 'faq')} goes back to the depot it came from.",
        f"If your delivery was {_register_word('late', 'faq')} because of a "
        f"{_register_word('delay', 'faq')}, you can ask for your "
        f"{_register_word('refund', 'faq')} within {stated_days} days.",
    ]
    return {
        "doc_id": f"faq-{key}" + ("-conflict" if contradicts else ""),
        "type": "faq",
        "title": f"FAQ: {title.lower()}",
        "effective": "2025-06-01",
        "superseded": False,
        "text": _para(rng, body),
        "facts": {"hold_hours": stated_hours, "claim_days": stated_days,
                  "topic": key, "authoritative": not contradicts},
    }


def depot_handbook(rng: random.Random, depot: str, section: int,
                   local_rule: str) -> dict:
    """Near-duplicates by construction: handbooks are ~90% shared text with one
    depot-specific paragraph, which is how real operational documentation
    actually looks."""
    shared = [
        f"Section {section}. Inbound processing at the {depot} "
        f"{_register_word('parcel', 'ops')} hub.",
        f"Every inbound {_register_word('parcel', 'ops')} is scanned against "
        f"the manifest on arrival and again when it leaves the sortation area.",
        f"An {_register_word('address_check', 'ops')} raised by the sorter "
        f"routes the unit to the exceptions bay rather than to the outbound lane.",
        "Dwell time in the exceptions bay is reported daily and any unit "
        "exceeding the threshold is escalated to the duty supervisor.",
    ]
    return {
        "doc_id": f"handbook-{depot}-s{section}",
        "type": "handbook",
        "title": f"{depot} depot handbook, section {section}",
        "effective": "2025-01-15",
        "superseded": False,
        "text": _para(rng, [*shared, local_rule]),
        "facts": {"depot": depot, "section": section},
    }


def incident_report(rng: random.Random, ref: str, depot: str, shipment: str,
                    date: str, cause: str) -> dict:
    body = [
        f"Incident {ref}, raised {date} at the {depot} depot.",
        f"Consignment {shipment} was held in the exceptions bay following an "
        f"{_register_word('address_check', 'ops')} raised during inbound scan.",
        f"Root cause: {cause}.",
        f"The {_register_word('parcel', 'ops')} was released after the delivery "
        f"point was confirmed by the customer service desk.",
        f"Total dwell was recorded against the {depot} exception report for "
        f"that week.",
    ]
    return {
        "doc_id": f"incident-{ref}",
        "type": "incident",
        "title": f"Incident {ref}",
        "effective": date,
        "superseded": False,
        "text": _para(rng, body),
        "facts": {"depot": depot, "shipment": shipment, "cause": cause,
                  "ref": ref, "date": date},
    }


def shipment_record(rng: random.Random, shipment: str, depot: str, carrier: str,
                    status: str) -> dict:
    body = [
        f"Consignment {shipment}.",
        f"Origin hub: {depot}. Carrier: {carrier}. Current status: {status}.",
        f"This consignment is handled under the operating procedures of the "
        f"{depot} depot.",
        "Scan history is retained for ninety days from the final movement.",
    ]
    return {
        "doc_id": f"shipment-{shipment}",
        "type": "shipment",
        "title": f"Consignment {shipment}",
        "effective": "2025-08-01",
        "superseded": False,
        "text": _para(rng, body),
        "facts": {"shipment": shipment, "depot": depot, "carrier": carrier,
                  "status": status},
    }


def carrier_agreement(rng: random.Random, carrier: str, clause: int) -> dict:
    body = [
        f"Clause {clause} of the carriage agreement with {carrier}.",
        f"The carrier shall deliver {_register_word('parcel', 'legal')} to the "
        f"delivery-point confirmation recorded on the manifest.",
        f"Where the carrier is {_register_word('late', 'legal')}, the customer "
        f"may seek {_register_word('refund', 'legal')} in accordance with the "
        f"published policy.",
        f"Nothing in this clause displaces the operator's own "
        f"{_register_word('address_check', 'legal')} procedures.",
    ]
    return {
        "doc_id": f"agreement-{carrier.split()[0].lower()}-c{clause}",
        "type": "agreement",
        "title": f"{carrier} carriage agreement, clause {clause}",
        "effective": "2024-11-01",
        "superseded": False,
        "text": _para(rng, body),
        "facts": {"carrier": carrier, "clause": clause},
    }


def release_note(rng: random.Random, version: str, date: str, change: str) -> dict:
    body = [
        f"Tracking system release {version}, deployed {date}.",
        f"Change: {change}.",
        "Operators should expect the exceptions bay report to reflect this "
        "from the next working day.",
        "No action is required at depot level.",
    ]
    return {
        "doc_id": f"release-{version}",
        "type": "release",
        "title": f"Release {version}",
        "effective": date,
        "superseded": False,
        "text": _para(rng, body),
        "facts": {"version": version, "date": date},
    }


# --- corpus assembly --------------------------------------------------------


def build(seed: int = SEED) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    docs: list[dict] = []
    queries: list[dict] = []

    # 1. Policies, half of them in two versions so "current" is a real question.
    policy_facts: dict[str, dict] = {}
    for topic in POLICY_TOPICS:
        key, title = topic
        two_versions = rng.random() < 0.5
        if two_versions:
            v2 = policy_doc(rng, topic, 2, "2025-04-01", None)
            v1 = policy_doc(rng, topic, 1, "2024-02-01", v2["doc_id"])
            docs += [v1, v2]
            policy_facts[key] = v2["facts"]
        else:
            v1 = policy_doc(rng, topic, 1, "2024-02-01", None)
            docs.append(v1)
            policy_facts[key] = v1["facts"]

    # 2. FAQs, some of which contradict the policy they paraphrase.
    for topic in POLICY_TOPICS:
        key, _ = topic
        f = policy_facts[key]
        contradicts = rng.random() < 0.6
        docs.append(faq_entry(rng, topic, f["hold_hours"], f["claim_days"], contradicts))

    # 3. Depot handbooks — near-duplicates with one local rule each.
    local_rules = {
        d: (f"Local rule: at {d}, units flagged for "
            f"{_register_word('address_check', 'ops')} are staged in bay "
            f"{rng.randrange(2, 9)} and reviewed at {rng.choice(['06:00', '14:00', '22:00'])}.")
        for d in DEPOTS
    }
    for depot in DEPOTS:
        for section in range(1, 9):
            docs.append(depot_handbook(rng, depot, section, local_rules[depot]))

    # 4. Shipments, incidents, agreements, releases — the bulk and the noise.
    shipments = [f"TL-{rng.randrange(1000, 9999)}" for _ in range(1200)]
    shipments[0] = "TL-4471"  # continuity with the Module 0 fixtures
    ship_depot: dict[str, str] = {}
    for s in shipments:
        depot = rng.choice(DEPOTS)
        ship_depot[s] = depot
        docs.append(shipment_record(rng, s, depot, rng.choice(CARRIERS),
                                    rng.choice(["in transit", "delivered", "held",
                                                "returned to origin"])))

    causes = ["incomplete delivery point", "recipient unavailable at scan time",
              "postcode mismatch against the address database",
              "flat number missing from the label",
              "carrier could not access the delivery point"]
    for i in range(900):
        s = rng.choice(shipments)
        docs.append(incident_report(rng, f"INC-{4000 + i}", ship_depot[s], s,
                                    f"2025-{rng.randrange(1, 13):02d}-"
                                    f"{rng.randrange(1, 29):02d}",
                                    rng.choice(causes)))

    for carrier in CARRIERS:
        for clause in range(1, 25):
            docs.append(carrier_agreement(rng, carrier, clause))

    changes = ["exceptions bay dwell is now reported in minutes rather than hours",
               "address verification holds appear on the depot dashboard",
               "manifest scans are deduplicated by consignment id",
               "the returns queue is split by carrier"]
    for i in range(120):
        docs.append(release_note(rng, f"2025.{i // 12 + 1}.{i % 12}",
                                 f"2025-{i % 12 + 1:02d}-05", rng.choice(changes)))

    # Pad to the target with additional shipment records, which is what a real
    # corpus of this kind is mostly made of.
    while len(docs) < TARGET_DOCS:
        s = f"TL-{rng.randrange(1000, 9999)}"
        depot = rng.choice(DEPOTS)
        ship_depot.setdefault(s, depot)
        docs.append(shipment_record(rng, s, depot, rng.choice(CARRIERS),
                                    rng.choice(["in transit", "delivered", "held"])))

    for d in docs:
        _expand(rng, d)
    for i, d in enumerate(docs):
        d["doc_index"] = i

    by_id = {d["doc_id"]: d for d in docs}
    queries = build_queries(rng, docs, by_id, policy_facts, ship_depot)
    return docs, queries


def build_queries(rng: random.Random, docs: list[dict], by_id: dict[str, dict],
                  policy_facts: dict, ship_depot: dict) -> list[dict]:
    """Queries written alongside the corpus, each with its gold document
    recorded at the moment the query is written.

    Seven phenomena, planted at known rates. `tools/verify_corpus.py` then
    measures that each one is actually present rather than trusting that it is.
    """
    q: list[dict] = []

    def add(text: str, gold: list[str], kind: str, note: str = "") -> None:
        q.append({"query_id": f"q{len(q):03d}", "text": text,
                  "gold_doc_ids": gold, "phenomenon": kind, "note": note})

    def current_policy(key: str) -> dict:
        return next(d for d in docs if d["facts"].get("topic") == key
                    and d["type"] == "policy" and not d["superseded"])

    # (a) Vocabulary mismatch: customer wording, operational answer. Three
    #     phrasings per topic, none of which shares the policy's own terms.
    faq_phrasings = [
        "how long do you keep my parcel when there is an address check on it",
        "can I get my money back if a hold-up made my parcel late",
        "what happens to my parcel if you cannot confirm where it is going",
    ]
    for key, title in POLICY_TOPICS:
        policy = current_policy(key)
        for phrasing in faq_phrasings:
            add(f"{phrasing} for {title.lower()}", [policy["doc_id"]],
                "vocabulary_mismatch",
                "query is in the customer register; the gold document is in the "
                "operational register and shares almost no vocabulary with it")

    # (b) Superseded versions: only the current one is gold, and v1 is a
    #     near-identical distractor.
    for key, facts in policy_facts.items():
        if facts["version"] == 2:
            current = by_id[f"policy-{key}-v2"]
            for phrasing in ("what is the current {} policy",
                             "which {} rules are in force now",
                             "what are the latest {} requirements",
                             "has the {} policy changed recently",
                             "which version of the {} policy applies today"):
                add(phrasing.format(key.replace("_", " ")), [current["doc_id"]],
                    "superseded", "v1 of the same policy is a near-identical distractor")

    # (c) Lexical-overlap distractors: the shipment id appears in the shipment
    #     record too, which is not the answer.
    incidents = [d for d in docs if d["type"] == "incident"]
    for inc in rng.sample(incidents, min(40, len(incidents))):
        s = inc["facts"]["shipment"]
        add(f"why was consignment {s} held", [inc["doc_id"]], "lexical_distractor",
            "the shipment record for the same id shares the rare term and is not "
            "the answer")

    # (d) Near-duplicates: five other depot handbooks are ~90% identical.
    for depot in DEPOTS:
        hb = next(d for d in docs if d["facts"].get("depot") == depot
                  and d["type"] == "handbook")
        for phrasing in ("which bay are address verification units staged in at the {}",
                         "what is the local staging rule at the {} depot",
                         "when are exception units reviewed at {}",
                         "what time is the exceptions bay checked at {}",
                         "where do flagged units go at the {} depot"):
            add(phrasing.format(depot), [hb["doc_id"]], "near_duplicate",
                "the other five depot handbooks are near-identical")

    # (e) Multi-hop: the shipment record gives the depot, the handbook gives
    #     the rule, and neither alone answers.
    shipments_with_records = [d for d in docs if d["type"] == "shipment"]
    for rec in rng.sample(shipments_with_records, min(40, len(shipments_with_records))):
        s = rec["facts"]["shipment"]
        depot = rec["facts"]["depot"]
        hb = next((d for d in docs if d["facts"].get("depot") == depot
                   and d["type"] == "handbook"), None)
        if hb:
            add(f"which bay would consignment {s} be staged in if it were flagged",
                [rec["doc_id"], hb["doc_id"]], "multi_hop",
                "needs the consignment's depot from one document and that depot's "
                "local rule from another")

    # (f) Contradictions: an FAQ states different numbers; the policy governs.
    for key, title in POLICY_TOPICS:
        faq = by_id.get(f"faq-{key}-conflict")
        if faq:
            policy = current_policy(key)
            for phrasing in ("authoritatively, how many hours is an address "
                             "verification hold for {}",
                             "what does the policy say the claim window is for {}",
                             "per the operational policy rather than the FAQ, what "
                             "is the hold duration for {}",
                             "which document governs the {} hold time, and what "
                             "does it state"):
                add(phrasing.format(title.lower()), [policy["doc_id"]],
                    "contradiction",
                    "an FAQ states a different number and is not authoritative")

    # (g) Unanswerable: abstention is the correct behaviour.
    unanswerable = [
        "what was the total revenue last quarter",
        "who is the current chief executive",
        "how many staff work at the scotland depot",
        "what is the fuel surcharge for international freight",
        "when does the new warehouse in cardiff open",
        "what is the carbon reduction target",
        "which insurer underwrites the fleet",
        "what is the staff discount on shipping",
        "how much did the tracking system cost to build",
        "what is the average tenure of a depot manager",
        "which depot has the lowest staff turnover",
        "what is the parental leave entitlement",
        "how many vehicles are in the fleet",
        "what is the notice period for a depot manager",
        "when is the next board meeting",
        "what is the policy on remote working",
        "how many parcels were delivered last year",
        "what is the on-time delivery rate for the scotland depot",
        "which carrier has the best damage record",
        "what does a pallet movement cost internally",
        "when was the tracking system last audited",
        "who signs off exceptions above a certain value",
        "what is the busiest week of the year",
        "how long are CCTV recordings retained",
    ]
    for text in unanswerable:
        add(text, [], "unanswerable",
            "no document supports this; abstention is the correct behaviour")

    # Balance to N_QUERIES: take proportionally from each phenomenon rather
    # than truncating a shuffled list, which would drop whole categories.
    by_kind: dict[str, list[dict]] = {}
    for item in q:
        by_kind.setdefault(item["phenomenon"], []).append(item)
    for group in by_kind.values():
        rng.shuffle(group)

    chosen: list[dict] = []
    kinds = sorted(by_kind)
    i = 0
    while len(chosen) < N_QUERIES and any(by_kind[k] for k in kinds):
        k = kinds[i % len(kinds)]
        if by_kind[k]:
            chosen.append(by_kind[k].pop())
        i += 1

    rng.shuffle(chosen)
    for n, item in enumerate(chosen):
        item["query_id"] = f"q{n:03d}"
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    docs, queries = build()
    kinds = Counter(d["type"] for d in docs)
    phen = Counter(q["phenomenon"] for q in queries)

    print(f"documents: {len(docs)}")
    for k, n in kinds.most_common():
        print(f"    {k:<12} {n:>5}")
    print(f"queries: {len(queries)}")
    for k, n in phen.most_common():
        print(f"    {k:<22} {n:>4}")

    chars = sum(len(d["text"]) for d in docs)
    print(f"\ntext: {chars / 1e6:.2f} M chars  (~{chars / 4 / 1e6:.2f} M tokens)")

    if args.stats:
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "documents.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with (OUT / "queries.jsonl").open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}/documents.jsonl and queries.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
