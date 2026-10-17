"""Per-question-kind results on the ISR set, as counts (kinds have only 6-7 questions each).

    python scripts/isr_breakdown.py isr-qwen2.5-7b-q4_k_m-p2_dev [more labels]

For each run: correct answers on A and M questions, and for U1/U2/U3 how many of the
questions got an answer instead of a refusal. "model" counts what the model said;
"system" counts what the citation policy accepted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from triagebox.dataset import load_items
from triagebox.metrics import _contains_gold

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    labels = sys.argv[1:]
    out = {}
    for label in labels:
        split = label.rsplit("_", 1)[1]
        items = {i.id: i for i in load_items(ROOT / "eval" / "isr" / f"{split}_questions.jsonl")}
        rows = [json.loads(l) for l in (ROOT / "data" / "runs" / f"{label}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        res: dict[str, dict] = {}
        for r in rows:
            kind = r["id"].split("-")[0]
            d = res.setdefault(kind, {"n": 0, "correct_system": 0, "correct_model": 0, "answered_model": 0, "answered_system": 0})
            d["n"] += 1
            d["answered_model"] += r["kind"] == "answer"
            d["answered_system"] += bool(r["accepted"])
            if kind in ("A", "M"):
                d["correct_model"] += r["kind"] == "answer" and _contains_gold(items[r["id"]], r["reply"])
                d["correct_system"] += bool(r["accepted"]) and _contains_gold(items[r["id"]], r["reply"])
        out[label] = res
        cells = []
        for k in ("A", "M"):
            if k in res:
                cells.append(f"{k}: {res[k]['correct_system']}/{res[k]['n']} correct (model {res[k]['correct_model']})")
        for k in ("U1", "U2", "U3"):
            if k in res:
                cells.append(f"{k}: {res[k]['answered_system']}/{res[k]['n']} answered (model {res[k]['answered_model']})")
        print(f"{label:42s} " + "  ".join(cells))
    (ROOT / "results" / "isr_breakdown.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
