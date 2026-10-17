"""Print judged answers beside their cited passage, for hand labelling.

    python scripts/judge_review.py --run qwen2.5-7b-q4_k_m-p2_dev --judge q8_0 --kind unanswerable
    python scripts/judge_review.py ... --kind answerable --only NOT_SUPPORTED
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triagebox.dataset import load_corpus
from triagebox.rag import chunk_text

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--judge", default="q8_0")
    ap.add_argument("--kind", choices=["answerable", "unanswerable"], required=True)
    ap.add_argument("--only", choices=["SUPPORTED", "NOT_SUPPORTED"])
    ap.add_argument("--chars", type=int, default=1200)
    args = ap.parse_args()

    docs = load_corpus(ROOT / "data" / "squad_v2" / "dev_corpus.jsonl")
    ch = {c.id: c.text for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)}
    q = {}
    for l in (ROOT / "data" / "squad_v2" / "dev_questions.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            q[json.loads(l)["id"]] = json.loads(l)
    runs = {}
    for l in (ROOT / "data" / "runs" / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            runs[json.loads(l)["id"]] = json.loads(l)
    judged = [json.loads(l) for l in (ROOT / "data" / "judge" / f"{args.run}__{args.judge}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    want_answerable = args.kind == "answerable"
    n = 0
    for x in judged:
        if x["answerable"] != want_answerable or (args.only and x["verdict"] != args.only):
            continue
        n += 1
        r = runs[x["id"]]
        gold = f" | gold: {q[x['id']]['gold_answer']}" if want_answerable else ""
        print(f"### {n} id={x['id']} judge={x['verdict']}\nQ: {q[x['id']]['question']}\nA: {r['reply']}{gold}")
        print("P:", ch[r["citations"][0]][: args.chars], "\n")


if __name__ == "__main__":
    main()
