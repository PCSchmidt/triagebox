"""Offline sweep of the lexical grounding policy over saved runs (no model needed).

    python scripts/grounding_policy.py

For each saved dev run, re-scores the system as if answers whose content words are
not found in their cited passages (support < threshold) had been refused. Reports
what that does to accuracy, false refusal and the dataset-labelled hallucination
rate. For the run with hand labels it also reports how many answers the passage
does not support were removed, and how many supported ones were lost.
"""

from __future__ import annotations

import json
from pathlib import Path

from triagebox.citations import lexical_support
from triagebox.dataset import load_corpus, load_items
from triagebox.metrics import Outcome, score
from triagebox.prompt import ParsedResponse
from triagebox.rag import chunk_text

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    "qwen2.5-1.5b-fp16-p2_dev",
    "qwen2.5-1.5b-q8_0-p2_dev",
    "qwen2.5-1.5b-q4_k_m-p2_dev",
    "qwen2.5-7b-q8_0-p2_dev",
    "qwen2.5-7b-q4_k_m-p2_dev",
]
THRESHOLDS = [None, 0.5, 0.7, 0.9, 1.0]


def main() -> None:
    docs = load_corpus(ROOT / "data" / "squad_v2" / "dev_corpus.jsonl")
    ch = {c.id: c.text for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)}
    items = {i.id: i for i in load_items(ROOT / "data" / "squad_v2" / "dev_questions.jsonl")}
    hand = json.loads((ROOT / "results" / "hand_labels_dev.json").read_text(encoding="utf-8"))["labels"]
    out = {}
    for run in RUNS:
        rows = [json.loads(l) for l in (ROOT / "data" / "runs" / f"{run}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        support = {}
        for r in rows:
            if r["kind"] == "answer" and r["accepted"]:
                support[r["id"]] = lexical_support(r["reply"], [ch[c] for c in r["citations"] if c in ch])
        out[run] = []
        for t in THRESHOLDS:
            outcomes, removed_unsup = [], 0
            kept_sup = lost_sup = removed_unsup = kept_unsup = 0
            for r in rows:
                item = items[r["id"]]
                parsed = ParsedResponse(r["kind"], r["reply"], tuple(r["citations"]))
                if not r["accepted"]:
                    parsed = ParsedResponse("refusal" if r["kind"] != "malformed" else "malformed", r["reply"])
                elif t is not None and support[r["id"]] < t:
                    parsed = ParsedResponse("refusal", "grounding policy")
                    if run.startswith("qwen2.5-7b-q4_k_m") and r["id"] in hand:
                        pass
                outcomes.append(Outcome(item, parsed, frozenset(r["retrieved"])))
                if run == "qwen2.5-7b-q4_k_m-p2_dev" and not item.answerable and r["id"] in hand and r["accepted"]:
                    gone = t is not None and support[r["id"]] < t
                    if hand[r["id"]] == "S":
                        lost_sup += gone
                        kept_sup += not gone
                    else:
                        removed_unsup += gone
                        kept_unsup += not gone
            s = score(outcomes)
            row = {
                "min_support": t,
                "accuracy": s.accuracy,
                "false_refusal_rate": s.false_refusal_rate,
                "hallucination_rate_dataset_labels": s.hallucination_rate,
            }
            if run == "qwen2.5-7b-q4_k_m-p2_dev":
                row["hand_unsupported_answers_removed_of_14"] = removed_unsup
                row["hand_supported_answers_lost_of_17"] = lost_sup
            out[run].append(row)
    (ROOT / "results" / "grounding_policy_dev.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for run, rows in out.items():
        print(run)
        for r in rows:
            extra = f"  hand: removed {r['hand_unsupported_answers_removed_of_14']}/14 unsupported, lost {r['hand_supported_answers_lost_of_17']}/17 supported" if "hand_supported_answers_lost_of_17" in r else ""
            print(f"  support>={r['min_support']}: acc {100*r['accuracy']:.0f}%  false-refusal {100*r['false_refusal_rate']:.0f}%  halluc {100*r['hallucination_rate_dataset_labels']:.0f}%{extra}")


if __name__ == "__main__":
    main()
