"""Evaluate detectors of unsupported answers on labelled answers from saved runs.

    python scripts/detector_eval.py --source isr --split dev
    python scripts/detector_eval.py --source squad --split dev      # hand-labelled 7B run

Labels (1 = supported, 0 = unsupported):
  ISR: a correct answer to an answerable question = 1; any accepted answer to an
       unanswerable question = 0 (true by construction). Wrong answers to answerable
       questions are left out: whether they are "unsupported" is not defined here.
  SQuAD (7B Q4_K_M, prompt v2): my hand labels for answers to unanswerable questions
       (results/hand_labels_dev.json); for answerable questions, the answer is
       supported if the gold string is in the cited passage and unsupported if the
       answer was right but the cited passage lacks it.

Detectors compared by AUROC (higher score = more likely supported): lexical overlap
of the answer with the cited passage; quote verification (found / score; see
triagebox.verify). The quote is produced by Qwen2.5-7B Q8_0 via llama.cpp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from triagebox.citations import lexical_support
from triagebox.dataset import load_corpus, load_items
from triagebox.llamacpp import LlamaServer
from triagebox.metrics import _contains_gold
from triagebox.rag import chunk_text
from triagebox.serverproc import ServerProc
from triagebox.verify import QUOTE_PROMPTS, clean_quote, verify
import re

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
MODEL = ROOT / "models" / "qwen2.5-7b-instruct" / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
DATASETS = {"squad": ROOT / "data" / "squad_v2", "isr": ROOT / "eval" / "isr"}
RUNS = {
    ("isr", "dev"): [f"isr-{m}_dev" for m in (
        "qwen2.5-7b-q4_k_m-p2", "qwen2.5-7b-q8_0-p2", "qwen2.5-1.5b-q8_0-p3",
        "qwen2.5-1.5b-q4_k_m-p3", "qwen2.5-1.5b-q8_0-p2", "qwen2.5-1.5b-q4_k_m-p2")],
    ("isr", "test"): [f"isr-{m}_test" for m in (
        "qwen2.5-7b-q4_k_m-p2", "qwen2.5-7b-q8_0-p2", "qwen2.5-1.5b-fp16-p3",
        "qwen2.5-1.5b-q8_0-p3", "qwen2.5-1.5b-q4_k_m-p3")],
    ("squad", "dev"): ["qwen2.5-7b-q4_k_m-p2_dev"],
}


def auroc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["isr", "squad"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--quote-prompt", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    suffix = "" if args.quote_prompt == 1 else f"_q{args.quote_prompt}"
    data = DATASETS[args.source]
    docs = load_corpus(data / f"{args.split}_corpus.jsonl")
    ch = {c.id: c.text for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)}
    items = {i.id: i for i in load_items(data / f"{args.split}_questions.jsonl")}
    hand = json.loads((ROOT / "results" / "hand_labels_dev.json").read_text(encoding="utf-8"))["labels"]

    todo = []
    for label in RUNS[(args.source, args.split)]:
        for l in (ROOT / "data" / "runs" / f"{label}.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if not (r["kind"] == "answer" and r["accepted"]):
                continue
            it = items[r["id"]]
            cited = [ch[c] for c in r["citations"] if c in ch]
            y = None
            if args.source == "isr":
                if not it.answerable:
                    y = 0
                elif _contains_gold(it, r["reply"]):
                    y = 1
            else:
                if r["id"] in hand and not it.answerable:
                    y = 1 if hand[r["id"]] == "S" else 0
                elif it.answerable and _contains_gold(it, r["reply"]):
                    y = 1 if any(_contains_gold(it, p) for p in cited) else 0
            if y is not None:
                todo.append((label, r, it, cited, y))
    print(f"{args.source} {args.split}: {len(todo)} labelled answers ({sum(t[4] for t in todo)} supported, {sum(1 - t[4] for t in todo)} unsupported)", flush=True)

    out_dir = ROOT / "data" / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"quotes_{args.source}_{args.split}{suffix}.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = [t for t in todo if f"{t[0]}|{t[1]['id']}" not in cache]
    if missing:
        with ServerProc(EXE, MODEL, log=ROOT / "data" / "logs" / "detector_server.log") as s:
            srv = LlamaServer(s.url, max_tokens=120, seed=0)
            for n, (label, r, it, cited, y) in enumerate(missing, 1):
                answer = re.sub(r"\s*\[[^\[\]\s]+\]", "", r["reply"]).strip()
                passages = "\n\n".join(f"[{cid}] {ch[cid]}" for cid in r["citations"] if cid in ch)
                cache[f"{label}|{r['id']}"] = srv.generate(QUOTE_PROMPTS[args.quote_prompt].format(passages=passages, question=it.question, answer=answer))
                if n % 40 == 0:
                    print(f"  {n}/{len(missing)}", flush=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")
    rows = []
    for label, r, it, cited, y in todo:
        reply = cache[f"{label}|{r['id']}"]
        answer = re.sub(r"\s*\[[^\[\]\s]+\]", "", r["reply"]).strip()
        sup = verify(it.question, answer, clean_quote(reply), cited)
        rows.append({
            "run": label, "id": r["id"], "label": y, "lexical": lexical_support(r["reply"], cited),
            "match": sup.match, "found": sup.found, "q_cov": sup.question_coverage, "e_cov": sup.entity_coverage,
            "a_cov": sup.answer_coverage, "quote": reply.strip()[:300],
        })

    out_dir = ROOT / "data" / "detector"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.source}_{args.split}{suffix}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    for r in rows:
        f = 1.0 if r["found"] else 0.0
        r["s_found"] = f
        r["s_found_ans"] = f * r["a_cov"]
        r["s_found_ent"] = f * r["e_cov"]
        r["s_found_min"] = f * min(r["a_cov"], r["e_cov"])
        r["s_match_min"] = r["match"] * min(r["a_cov"], r["e_cov"])
        r["s_ent_only"] = r["e_cov"]
    (out_dir / f"{args.source}_{args.split}{suffix}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    y = [r["label"] for r in rows]
    detectors = {
        "lexical_overlap": "lexical", "entity_coverage_only": "s_ent_only", "quote_found": "s_found",
        "found_x_answer_coverage": "s_found_ans", "found_x_entity_coverage": "s_found_ent",
        "found_x_min(answer,entity)": "s_found_min", "match_x_min(answer,entity)": "s_match_min",
    }
    summary = {
        "source": args.source, "split": args.split, "n": len(rows), "supported": sum(y), "unsupported": len(y) - sum(y),
        "auroc": {name: auroc([r[key] for r in rows], y) for name, key in detectors.items()},
    }
    for name, key in (("lexical_overlap", "lexical"), ("entity_coverage_only", "s_ent_only"), ("found_x_min(answer,entity)", "s_found_min"), ("match_x_min(answer,entity)", "s_match_min")):
        pts = []
        for t in (0.3, 0.5, 0.7, 0.9, 1.0):
            kept = [r for r in rows if r[key] >= t]
            pts.append({
                "threshold": t,
                "supported_kept": round(sum(r["label"] for r in kept) / max(1, sum(y)), 3),
                "unsupported_removed": round(1 - sum(1 - r["label"] for r in kept) / max(1, len(y) - sum(y)), 3),
            })
        summary[f"operating_points_{name}"] = pts
    (ROOT / "results" / f"detector_{args.source}_{args.split}{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
