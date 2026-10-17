"""Compare judge verdicts with hand labels and with a deterministic check.

    python scripts/judge_agreement.py --run qwen2.5-7b-q4_k_m-p2_dev --judge q8_0

* Hand labels (results/hand_labels_dev.json): partial, see its "scope" field.
* Deterministic check, answerable questions only: is the gold answer (or an
  alternative) a substring of the cited passage after normalisation? If not, the
  answer is not grounded in what it cited, whatever the judge says.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.metrics import _contains_gold
from triagebox.rag import chunk_text

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", default="q8_0")
    args = ap.parse_args()

    docs = load_corpus(ROOT / "data" / "squad_v2" / "dev_corpus.jsonl")
    ch = {c.id: c.text for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)}
    items = {i.id: i for i in load_items(ROOT / "data" / "squad_v2" / "dev_questions.jsonl")}
    runs = {r["id"]: r for r in map(json.loads, (ROOT / "data" / "runs" / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines())}
    judged = [json.loads(l) for l in (ROOT / "data" / "judge" / f"{args.run}__{args.judge}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    hand = json.loads((ROOT / "results" / "hand_labels_dev.json").read_text(encoding="utf-8"))

    # 1. judge vs hand labels
    tp = tn = fp = fn = 0  # positive = SUPPORTED
    for x in judged:
        h = hand["labels"].get(x["id"])
        if h is None or x["verdict"] == "UNPARSED":
            continue
        j, hs = x["verdict"] == "SUPPORTED", h == "S"
        tp += j and hs
        tn += (not j) and (not hs)
        fp += j and (not hs)
        fn += (not j) and hs
    n = tp + tn + fp + fn

    # 2. judge vs deterministic gold-in-cited-passage check (answerable only)
    ctp = ctn = cfp = cfn = 0
    for x in judged:
        if not x["answerable"]:
            continue
        r = runs[x["id"]]
        grounded = any(_contains_gold(items[x["id"]], ch[c]) for c in r["citations"] if c in ch)
        j = x["verdict"] == "SUPPORTED"
        ctp += j and grounded
        ctn += (not j) and (not grounded)
        cfp += j and (not grounded)
        cfn += (not j) and grounded
    cn = ctp + ctn + cfp + cfn

    # hand-label tallies on the unanswerable questions that were answered
    un = [x for x in judged if not x["answerable"]]
    un_supported = sum(hand["labels"][x["id"]] == "S" for x in un)
    answerable_total = sum(1 for x in judged if x["answerable"])

    out = {
        "run": args.run,
        "judge": args.judge,
        "vs_hand_labels": {
            "n": n,
            "agreement": (tp + tn) / n,
            "judge_supported_hand_not": fp,
            "judge_not_supported_hand_supported": fn,
            "both_supported": tp,
            "both_not_supported": tn,
            "caveat": "43 items, not a random sample (all answered-unanswerable plus the judge's rejections of answerable ones); single labeller; 4 ambiguous items listed in hand_labels_dev.json",
        },
        "vs_gold_in_cited_passage_answerable": {
            "n": cn,
            "agreement": (ctp + ctn) / cn,
            "judge_supported_but_gold_absent_from_cited": cfp,
            "judge_rejected_but_gold_in_cited": cfn,
            "both_supported": ctp,
            "both_not": ctn,
        },
        "answered_unanswerable_by_hand": {
            "n_answered": len(un),
            "passage_states_the_answer": un_supported,
            "passage_does_not": len(un) - un_supported,
        },
        "answerable_answered_with_gold_absent_from_cited_passage": ctn + cfp,
        "answerable_answered_total": answerable_total,
    }
    (ROOT / "results" / f"judge_agreement_{args.run}__{args.judge}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
