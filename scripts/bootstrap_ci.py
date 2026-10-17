"""Bootstrap 95% intervals (over questions) for the system-level dev metrics of saved runs.

    python scripts/bootstrap_ci.py [label ...]

Re-running at temperature 0 gives the same answers, so run-to-run repeats say little;
the uncertainty that matters is which 100 + 100 questions were drawn. Questions are
resampled with replacement within each kind (2,000 resamples, fixed seed).
System level: an answer counts only if it was accepted by the citation policy.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from triagebox.dataset import load_items
from triagebox.metrics import _contains_gold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = [
    "qwen2.5-1.5b-fp16-p2_dev", "qwen2.5-1.5b-q8_0-p2_dev", "qwen2.5-1.5b-q4_k_m-p2_dev",
    "qwen2.5-7b-q8_0-p2_dev", "qwen2.5-7b-q4_k_m-p2_dev",
]


def interval(flags: list[int], resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(flags)
    stats = sorted(sum(flags[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(flags) / n, stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]


def main() -> None:
    labels = [a if a.endswith("_dev") else a + "_dev" for a in sys.argv[1:]] or DEFAULT
    items = {i.id: i for i in load_items(ROOT / "data" / "squad_v2" / "dev_questions.jsonl")}
    out = {}
    for label in labels:
        rows = [json.loads(l) for l in (ROOT / "data" / "runs" / f"{label}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        ans = [r for r in rows if r["answerable"]]
        una = [r for r in rows if not r["answerable"]]
        acc = [int(r["accepted"] and _contains_gold(items[r["id"]], r["reply"])) for r in ans]
        halluc = [int(r["accepted"]) for r in una]
        refuse = [int(not r["accepted"]) for r in ans]
        out[label] = {name: dict(zip(("estimate", "low", "high"), (round(x, 3) for x in interval(f)))) for name, f in
                      (("accuracy", acc), ("answerable_not_answered", refuse), ("hallucination", halluc))}
        o = out[label]
        print(label, "  ".join(f"{k} {100*v['estimate']:.0f}% [{100*v['low']:.0f}-{100*v['high']:.0f}]" for k, v in o.items()))
    (ROOT / "results" / "bootstrap_ci_dev.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
