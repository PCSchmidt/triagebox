"""Run the TriageBox pipeline on the benchmark against a local llama-server and score it.

    llama-server -m <model.gguf> -ngl 99 -c 4096 --port 8080 --seed 0
    python scripts/run_eval.py --label qwen2.5-7b-q4_k_m --split dev --n-per-kind 100 --power

Reports two levels of metrics:

* model-level: what the model said (answered or refused), before any policy;
* system-level: what TriageBox accepts, i.e. answers that pass the citation policy.
  A rejected answer counts as a refusal.

Only aggregates are written under ``results/`` (tracked). Per-question replies go
to ``data/runs/`` (untracked) because they quote the benchmark's passages.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from triagebox.audit import AuditLog
from triagebox.dataset import load_corpus, load_items
from triagebox.llamacpp import LlamaServer, TimedLLM
from triagebox.metrics import Outcome, score
from triagebox.pipeline import triage
from triagebox.power import GpuPower, THROTTLE_BITS, timing_trustworthy
from triagebox.prompt import ParsedResponse
from triagebox.rag import Bm25Index, chunk_text

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"squad": ROOT / "data" / "squad_v2", "isr": ROOT / "eval" / "isr"}


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def med(xs: list[float]) -> float | None:
    return statistics.median(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server-url", default="http://127.0.0.1:8080")
    ap.add_argument("--label", required=True, help="e.g. qwen2.5-7b-q4_k_m")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="squad")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n-per-kind", type=int, default=100)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--prompt-version", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--power", action="store_true", help="sample GPU power with nvidia-smi")
    args = ap.parse_args()
    if args.split == "test":
        print("NOTE: the test split is for final numbers only; do not tune on it.", file=sys.stderr)

    DATA = DATASETS[args.dataset]
    docs = load_corpus(DATA / f"{args.split}_corpus.jsonl")
    all_items = load_items(DATA / f"{args.split}_questions.jsonl")
    ans = [i for i in all_items if i.answerable][: args.n_per_kind]
    unans = [i for i in all_items if not i.answerable][: args.n_per_kind]
    items = [x for pair in itertools.zip_longest(ans, unans) for x in pair if x is not None]  # interleave so drift hits both kinds equally
    chunks = [c for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)]
    index = Bm25Index(chunks)

    server = LlamaServer(args.server_url, max_tokens=args.max_tokens, seed=0)
    if not server.health():
        raise SystemExit(f"no healthy llama-server at {args.server_url}")
    llm = TimedLLM(server)
    for _ in range(2):  # warm-up, discarded
        server.generate("Say OK.")
    audit = AuditLog()

    model_out, system_out, rows = [], [], []
    ceiling_hits = 0
    power = GpuPower() if args.power else None
    run_t0 = time.perf_counter()
    if power:
        power.__enter__()
    try:
        for n, item in enumerate(items, 1):
            res = triage(item.question, index, llm, k=args.k, audit=audit, ts=datetime.now(timezone.utc).isoformat(), prompt_version=args.prompt_version)
            start, end, timing = llm.timings[-1]
            watts = power.mean_watts(start, end) if power else None
            joules = watts * timing.total_ms / 1000.0 if watts is not None else None
            retrieved = frozenset(res.retrieved_ids)
            model_out.append(Outcome(item, res.parsed, retrieved))
            sys_parsed = res.parsed if res.accepted else ParsedResponse("refusal", res.reason)
            system_out.append(Outcome(item, sys_parsed, retrieved))
            if item.answerable and all(g in {r.rsplit("#", 1)[0] for r in res.retrieved_ids} for g in item.gold_docs):
                ceiling_hits += 1
            rows.append(
                {
                    "id": item.id,
                    "answerable": item.answerable,
                    "kind": res.parsed.kind,
                    "accepted": res.accepted,
                    "reason": res.reason,
                    "raw": audit.entries[-1]["event"]["raw_reply"],
                    "reply": res.parsed.text,
                    "citations": list(res.parsed.citations),
                    "retrieved": list(res.retrieved_ids),
                    "ttft_ms": timing.ttft_ms,
                    "total_ms": timing.total_ms,
                    "tokens": timing.n_tokens,
                    "tokens_per_s": timing.tokens_per_s,
                    "watts": watts,
                    "joules": joules,
                }
            )
            if n % 20 == 0:
                print(f"  {n}/{len(items)}", flush=True)
    finally:
        if power:
            power.__exit__(None, None, None)

    run_t1 = time.perf_counter()
    gpu = power.state(run_t0, run_t1) if power else None
    trustworthy = timing_trustworthy(gpu) if gpu else None
    n_answerable = sum(i.answerable for i in items)
    summary = {
        "label": args.label,
        "split": args.split,
        "n_answerable": n_answerable,
        "n_unanswerable": len(items) - n_answerable,
        "k": args.k,
        "prompt_version": args.prompt_version,
        "retrieval_ceiling_answerable": ceiling_hits / n_answerable if n_answerable else None,
        "model_level": asdict(score(model_out)),
        "system_level": asdict(score(system_out)),
        "gpu_state_during_run": None
        if gpu is None
        else {
            "median_sm_mhz": gpu.median_sm_mhz,
            "max_sm_mhz": gpu.max_sm_mhz,
            "clock_fraction": gpu.clock_fraction,
            "max_temp_c": gpu.max_temp_c,
            "power_or_thermal_throttle_seen": bool(gpu.throttle_reasons_seen & THROTTLE_BITS),
            "timing_trustworthy": trustworthy,
        },
        "timing": {
            "ttft_ms_median": med([r["ttft_ms"] for r in rows]),
            "tokens_per_s_median": med([r["tokens_per_s"] for r in rows if r["tokens"] > 1]),
            "total_ms_median": med([r["total_ms"] for r in rows]),
            "tokens_median": med([r["tokens"] for r in rows]),
            "joules_per_answer_median": med([r["joules"] for r in rows if r["joules"] is not None]),
            "gpu_watts_median": med([r["watts"] for r in rows if r["watts"] is not None]),
        },
        "meta": {
            "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "triagebox_commit": git_commit(),
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "retrieval": f"BM25, {args.k} chunks of up to 200 words",
            "energy": "GPU board power from nvidia-smi (100 ms samples) x duration; GPU only, not system power"
            if args.power
            else "not measured",
            "note": "single run; timing from streamed generation on a desktop GPU, not an edge device",
        },
    }

    (ROOT / "data" / "runs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "data" / "runs" / f"{args.label}_{args.split}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    (ROOT / "results" / f"{args.label}_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    print(f"\n{args.label} on {args.split}: {n_answerable} answerable + {len(items) - n_answerable} unanswerable, k={args.k}")
    print(f"retrieval ceiling (gold paragraph in top {args.k}): {pct(summary['retrieval_ceiling_answerable'])}")
    for level in ("model_level", "system_level"):
        s = summary[level]
        print(
            f"{level:13s} accuracy {pct(s['accuracy'])}  hallucination {pct(s['hallucination_rate'])}  "
            f"false-refusal {pct(s['false_refusal_rate'])}  citation-validity {pct(s['citation_validity'])} "
            f"(paragraph-level {pct(s['citation_validity_doc_level'])})  "
            f"malformed {pct(s['malformed_rate'])}"
        )
    t = summary["timing"]
    if gpu is not None:
        print(
            f"GPU during run: median SM clock {gpu.median_sm_mhz:.0f} of {gpu.max_sm_mhz} MHz ({100 * gpu.clock_fraction:.0f}%), "
            f"max {gpu.max_temp_c} C, throttled: {bool(gpu.throttle_reasons_seen & THROTTLE_BITS)} -> "
            f"timing {'TRUSTWORTHY' if trustworthy else 'NOT TRUSTWORTHY (speed/energy below reflect throttling)'}"
        )
    print(f"timing: ttft {t['ttft_ms_median']:.0f} ms, {t['tokens_per_s_median'] or 0:.1f} tok/s, {t['tokens_median']:.0f} tokens/answer")
    if t["joules_per_answer_median"] is not None:
        print(f"energy: {t['joules_per_answer_median']:.1f} J/answer (GPU board power ~{t['gpu_watts_median']:.0f} W)")


if __name__ == "__main__":
    main()
