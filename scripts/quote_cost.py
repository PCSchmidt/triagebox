"""What does the quote check cost per question in time and GPU energy?

    python scripts/quote_cost.py --n-per-kind 20 --rounds 2

Two servers run at once: the answerer (1.5B Q8_0 or 7B Q4_K_M) and the extractor (Qwen2.5-7B Q8_0, the same model
the quote-policy numbers in docs/detector.md used). For each round the GPU is cooled first, then two bursts run over
the same ISR dev questions:
  A  triage with the citation policy only;
  B  triage with the quote check as well (one extra extractor call per answer that passed the citation policy).
Time and energy are measured per burst (GPU board power integrated over the burst, as in scripts/ladder.py); a burst
with no thermal or hardware throttling that stayed at or below 85 C counts as steady (operation at the board's
power limit is tolerated and recorded, as in scripts/ladder.py). The difference B - A is the cost of the check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.llamacpp import LlamaServer, TimedLLM
from triagebox.pipeline import QuoteCheck, triage
from triagebox.power import HARD_THROTTLE_BITS, POWER_CAP_BIT, GpuPower, energy_joules, timing_steady, wait_for_cool
from triagebox.rag import Bm25Index, chunk_text
from triagebox.serverproc import ServerProc

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
EXTRACTOR = ROOT / "models" / "qwen2.5-7b-instruct" / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"
ANSWERERS = {
    "qwen2.5-1.5b-q8_0": ROOT / "models" / "qwen2.5-1.5b-instruct" / "qwen2.5-1.5b-instruct-q8_0.gguf",
    "qwen2.5-7b-q4_k_m": ROOT / "models" / "qwen2.5-7b-instruct" / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
}
PROMPT_VERSION = {"qwen2.5-1.5b-q8_0": 3, "qwen2.5-7b-q4_k_m": 2}  # as in the accuracy runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-kind", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cool-c", type=int, default=55)
    ap.add_argument("--cool-timeout", type=float, default=600)
    args = ap.parse_args()

    data = ROOT / "eval" / "isr"
    docs = load_corpus(data / "dev_corpus.jsonl")
    index = Bm25Index([c for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)])
    items = load_items(data / "dev_questions.jsonl")
    answerable = [i for i in items if i.answerable][: args.n_per_kind]
    unanswerable = [i for i in items if not i.answerable][: args.n_per_kind]
    questions = answerable + unanswerable
    out = {"n_questions": len(questions), "rounds": []}

    (ROOT / "data" / "logs").mkdir(parents=True, exist_ok=True)
    with ServerProc(EXE, EXTRACTOR, port=8082, log=ROOT / "data" / "logs" / "cost_extractor.log") as ext:
        ext_client = LlamaServer(ext.url, max_tokens=120, seed=0)
        for name, model in ANSWERERS.items():
            with ServerProc(EXE, model, port=8081, log=ROOT / "data" / "logs" / f"cost_{name}.log") as ans:
                client = LlamaServer(ans.url, max_tokens=256, seed=0)
                for _ in range(3):  # warm-up
                    client.generate("Say OK.")
                    ext_client.generate("Say OK.")
                for rnd in range(args.rounds):
                    row = {"answerer": name, "round": rnd}
                    for label, with_check in (("A_citation_only", False), ("B_with_quote_check", True)):
                        cool = wait_for_cool(args.cool_c, timeout_s=args.cool_timeout)
                        llm, ext_llm = TimedLLM(client), TimedLLM(ext_client)
                        qc = QuoteCheck(ext_llm) if with_check else None
                        accepted = 0
                        with GpuPower() as power:
                            t0 = time.perf_counter()
                            for q in questions:
                                r = triage(q.question, index, llm, k=4, prompt_version=PROMPT_VERSION[name], quote_check=qc)
                                accepted += r.accepted
                            t1 = time.perf_counter()
                            state = power.state(t0, t1)
                            joules = energy_joules(power.samples, t0, t1)
                        row[label] = {
                            "seconds": t1 - t0,
                            "ms_per_question": 1000 * (t1 - t0) / len(questions),
                            "joules_per_question": None if joules is None else joules / len(questions),
                            "answerer_calls": len(llm.timings),
                            "extractor_calls": len(ext_llm.timings),
                            "extractor_tokens_median": statistics.median([t.n_tokens for _, _, t in ext_llm.timings]) if ext_llm.timings else None,
                            "accepted": accepted,
                            "start_temp_c": cool.temp_c,
                            "max_temp_c": state.max_temp_c,
                            "clock_fraction": state.clock_fraction,
                            "power_capped": bool(state.throttle_reasons_seen & POWER_CAP_BIT),
                            "hard_throttle": bool(state.throttle_reasons_seen & HARD_THROTTLE_BITS),
                            "trustworthy": bool(timing_steady(state)),  # no thermal or hardware throttling, at most 85 C; a software power cap is tolerated and recorded
                        }
                        print(f"{name} round {rnd} {label}: {row[label]['ms_per_question']:.0f} ms/q, {row[label]['joules_per_question']} J/q, accepted {accepted}, extractor calls {row[label]['extractor_calls']}, trustworthy {row[label]['trustworthy']}", flush=True)
                    out["rounds"].append(row)
    (ROOT / "results" / "quote_cost.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nSummary (medians over trustworthy rounds)")
    for name in ANSWERERS:
        rs = [r for r in out["rounds"] if r["answerer"] == name and r["A_citation_only"]["trustworthy"] and r["B_with_quote_check"]["trustworthy"]]
        if not rs:
            print(f"{name}: no trustworthy round")
            continue
        m = lambda lab, k: statistics.median([r[lab][k] for r in rs])  # noqa: E731
        a_ms, b_ms = m("A_citation_only", "ms_per_question"), m("B_with_quote_check", "ms_per_question")
        a_j, b_j = m("A_citation_only", "joules_per_question"), m("B_with_quote_check", "joules_per_question")
        print(f"{name}: {a_ms:.0f} -> {b_ms:.0f} ms/question (x{b_ms / a_ms:.2f}); {a_j:.1f} -> {b_j:.1f} J/question (x{b_j / a_j:.2f}); extractor calls {m('B_with_quote_check', 'extractor_calls'):.0f} of {len(questions)}")


if __name__ == "__main__":
    main()
