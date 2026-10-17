"""Re-score saved runs under the quote-verification policy.

    python scripts/quote_policy.py --dataset squad --split dev
    python scripts/quote_policy.py --dataset isr --split dev
    python scripts/quote_policy.py --dataset isr --split test      # once; the policy has no tuned parameter

Policy: a second model call (Qwen2.5-7B Q8_0) is asked to copy the sentence of the cited passage(s)
that states the answer; the answer is kept only if at least 90% of that quote occurs as one
verbatim run in a cited passage (triagebox.verify.quote_match). 0.9 was fixed before any
results were seen and is not tuned. Rejected answers become refusals. Quotes are cached in
data/detector/ so a rerun costs nothing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.llamacpp import LlamaServer
from triagebox.metrics import Outcome, score
from triagebox.prompt import ParsedResponse
from triagebox.rag import chunk_text
from triagebox.serverproc import ServerProc
from triagebox.verify import QUOTE_PROMPTS, clean_quote, quote_match

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
MODEL = ROOT / "models" / "qwen2.5-7b-instruct" / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
DATASETS = {"squad": ROOT / "data" / "squad_v2", "isr": ROOT / "eval" / "isr"}
RUNS = {
    ("squad", "dev"): ["qwen2.5-1.5b-fp16-p3_dev", "qwen2.5-1.5b-q8_0-p3_dev", "qwen2.5-1.5b-q4_k_m-p3_dev", "qwen2.5-7b-q8_0-p2_dev", "qwen2.5-7b-q4_k_m-p2_dev"],
    ("isr", "dev"): ["isr-qwen2.5-1.5b-q8_0-p3_dev", "isr-qwen2.5-1.5b-q4_k_m-p3_dev", "isr-qwen2.5-7b-q8_0-p2_dev", "isr-qwen2.5-7b-q4_k_m-p2_dev"],
    ("isr", "test"): ["isr-qwen2.5-1.5b-fp16-p3_test", "isr-qwen2.5-1.5b-q8_0-p3_test", "isr-qwen2.5-1.5b-q4_k_m-p3_test", "isr-qwen2.5-7b-q8_0-p2_test", "isr-qwen2.5-7b-q4_k_m-p2_test"],
}
THRESHOLD = 0.9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["squad", "isr"], required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--quote-prompt", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    suffix = "" if args.quote_prompt == 1 else f"_q{args.quote_prompt}"
    data = DATASETS[args.dataset]
    docs = load_corpus(data / f"{args.split}_corpus.jsonl")
    ch = {c.id: c.text for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)}
    items = {i.id: i for i in load_items(data / f"{args.split}_questions.jsonl")}
    hand = json.loads((ROOT / "results" / "hand_labels_dev.json").read_text(encoding="utf-8"))["labels"]
    cache_dir = ROOT / "data" / "detector"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"quotes_{args.dataset}_{args.split}{suffix}.json"  # shared with detector_eval.py
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    runs = {run: [json.loads(l) for l in (ROOT / "data" / "runs" / f"{run}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] for run in RUNS[(args.dataset, args.split)]}
    missing = [(run, r) for run, rows in runs.items() for r in rows if r["kind"] == "answer" and r["accepted"] and f"{run}|{r['id']}" not in cache]
    print(f"{len(missing)} quotes to generate", flush=True)
    if missing:
        with ServerProc(EXE, MODEL, log=ROOT / "data" / "logs" / "detector_server.log") as s:
            srv = LlamaServer(s.url, max_tokens=120, seed=0)
            for n, (run, r) in enumerate(missing, 1):
                answer = re.sub(r"\s*\[[^\[\]\s]+\]", "", r["reply"]).strip()
                passages = "\n\n".join(f"[{cid}] {ch[cid]}" for cid in r["citations"] if cid in ch)
                cache[f"{run}|{r['id']}"] = srv.generate(QUOTE_PROMPTS[args.quote_prompt].format(passages=passages, question=items[r["id"]].question, answer=answer))
                if n % 50 == 0:
                    print(f"  {n}/{len(missing)}", flush=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    out = {}
    for run, rows in runs.items():
        res = []
        for policy in (False, True):
            outcomes, hand_removed, hand_lost = [], 0, 0
            for r in rows:
                parsed = ParsedResponse(r["kind"], r["reply"], tuple(r["citations"]))
                if not r["accepted"]:
                    parsed = ParsedResponse("refusal" if r["kind"] != "malformed" else "malformed", r["reply"])
                elif policy:
                    quote = clean_quote(cache[f"{run}|{r['id']}"])
                    ok = quote is not None and quote_match(quote, [ch[c] for c in r["citations"] if c in ch]) >= THRESHOLD
                    if not ok:
                        parsed = ParsedResponse("refusal", "quote policy")
                        if run == "qwen2.5-7b-q4_k_m-p2_dev" and r["id"] in hand and not items[r["id"]].answerable:
                            hand_removed += hand[r["id"]] == "N"
                            hand_lost += hand[r["id"]] == "S"
                outcomes.append(Outcome(items[r["id"]], parsed, frozenset(r["retrieved"])))
            s = score(outcomes)
            row = {"quote_policy": policy, "accuracy": s.accuracy, "false_refusal_rate": s.false_refusal_rate, "hallucination_rate": s.hallucination_rate}
            if run == "qwen2.5-7b-q4_k_m-p2_dev" and policy:
                row["hand_unsupported_removed_of_14"], row["hand_supported_lost_of_17"] = hand_removed, hand_lost
            res.append(row)
        out[run] = res
    (ROOT / "results" / f"quote_policy_{args.dataset}_{args.split}{suffix}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for run, res in out.items():
        print(run)
        for r in res:
            extra = f"  hand: removed {r['hand_unsupported_removed_of_14']}/14 unsupported, lost {r['hand_supported_lost_of_17']}/17 supported" if "hand_supported_lost_of_17" in r else ""
            print(f"  {'quote policy' if r['quote_policy'] else 'no policy   '}: acc {100 * r['accuracy']:.1f}%  false-refusal {100 * r['false_refusal_rate']:.1f}%  halluc {100 * r['hallucination_rate']:.1f}%{extra}")


if __name__ == "__main__":
    main()
