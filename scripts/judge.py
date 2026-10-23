"""Judge whether each accepted answer is supported by the passages it cites.

    python scripts/judge.py --run qwen2.5-7b-q4_k_m-p2_dev --judge q8_0

Reads the per-question replies saved by run_eval.py (data/runs/, untracked), shows
a judge model the question, the answer and ONLY the cited passages, and asks for
SUPPORTED or NOT_SUPPORTED. Writes per-item verdicts to data/judge/ (untracked,
quotes the benchmark) and aggregates to results/judge_<run>.json (tracked).

The judge is a local Qwen2.5-7B, the same family as the model being judged. That
is a known weakness (shared blind spots), which is why agreement with hand labels
is measured separately by scripts/judge_agreement.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from triagebox.dataset import load_corpus
from triagebox.judge import JUDGE_PROMPT, parse_verdict
from triagebox.llamacpp import LlamaServer
from triagebox.rag import chunk_text
from triagebox.serverproc import ServerProc

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
MODELS = {
    "q4_k_m": ROOT / "models" / "qwen2.5-7b-instruct" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
    "q8_0": ROOT / "models" / "qwen2.5-7b-instruct" / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf",
    # a judge from a different model family (Microsoft Phi-3-mini 4k instruct, 3.8B, MIT licence, 4-bit)
    "phi3": ROOT / "models" / "phi-3-mini" / "Phi-3-mini-4k-instruct-q4.gguf",
}
LABEL = {"q4_k_m": "Qwen2.5-7B-Instruct q4_k_m", "q8_0": "Qwen2.5-7B-Instruct q8_0", "phi3": "Phi-3-mini-4k-instruct q4"}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="label_split of a saved run, e.g. qwen2.5-7b-q4_k_m-p2_dev")
    ap.add_argument("--judge", choices=sorted(MODELS), default="q8_0")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    docs = load_corpus(ROOT / "data" / "squad_v2" / f"{args.split}_corpus.jsonl")
    chunk_text_by_id = {c.id: c.text for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)}
    questions = {
        json.loads(l)["id"]: json.loads(l)["question"]
        for l in (ROOT / "data" / "squad_v2" / f"{args.split}_questions.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    }
    rows = [json.loads(l) for l in (ROOT / "data" / "runs" / f"{args.run}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    answered = [r for r in rows if r["kind"] == "answer" and r["accepted"]]
    print(f"{len(answered)} accepted answers to judge with {LABEL[args.judge]}")

    out_rows = []
    with ServerProc(EXE, MODELS[args.judge], log=ROOT / "data" / "logs" / "judge_server.log") as s:
        srv = LlamaServer(s.url, max_tokens=60, seed=0)
        for n, r in enumerate(answered, 1):
            passages = "\n\n".join(f"[{cid}] {chunk_text_by_id[cid]}" for cid in r["citations"] if cid in chunk_text_by_id)
            reply = srv.generate(
                JUDGE_PROMPT.format(
                    passages=passages,
                    question=questions[r["id"]],
                    answer=re.sub(r"\s*\[[^\[\]\s]+\]", "", r["reply"]).strip(),
                )
            )
            out_rows.append({"id": r["id"], "answerable": r["answerable"], "verdict": parse_verdict(reply), "judge_reply": reply.strip()})
            if n % 25 == 0:
                print(f"  {n}/{len(answered)}", flush=True)

    d = ROOT / "data" / "judge"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{args.run}__{args.judge}.jsonl").write_text("\n".join(json.dumps(x) for x in out_rows) + "\n", encoding="utf-8")

    def rate(rs, v):
        return sum(x["verdict"] == v for x in rs) / len(rs) if rs else None

    a = [x for x in out_rows if x["answerable"]]
    u = [x for x in out_rows if not x["answerable"]]
    summary = {
        "run": args.run,
        "judge": LABEL[args.judge],
        "n_judged": len(out_rows),
        "answered_answerable": {"n": len(a), "supported": rate(a, "SUPPORTED"), "not_supported": rate(a, "NOT_SUPPORTED"), "unparsed": rate(a, "UNPARSED")},
        "answered_unanswerable": {"n": len(u), "supported": rate(u, "SUPPORTED"), "not_supported": rate(u, "NOT_SUPPORTED"), "unparsed": rate(u, "UNPARSED")},
        "note": "judge verdicts only; agreement with hand labels is in judge_agreement_<run>.json",
    }
    (ROOT / "results" / f"judge_{args.run}__{args.judge}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
