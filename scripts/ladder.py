"""The quantisation ladder: accuracy for every configuration, then thermally gated timing.

    python scripts/ladder.py accuracy            # one full dev evaluation per configuration
    python scripts/ladder.py timing --rounds 3   # short, cooled, interleaved timing bursts

Accuracy does not depend on GPU speed, so each configuration is evaluated once,
whatever the GPU is doing. Timing does, so it is measured differently: before
every burst the GPU is cooled to a set temperature, a burst is a few requests, the
configurations are rotated through several rounds (restarting the server each
time), and a burst only counts if the GPU held its clock and reported no power or
thermal throttling. Rejected bursts are reported, not hidden.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.llamacpp import LlamaServer, TimedLLM
from triagebox.pipeline import triage
from triagebox.power import (
    HARD_THROTTLE_BITS,
    POWER_CAP_BIT,
    GpuPower,
    energy_joules,
    timing_steady,
    timing_trustworthy,
    wait_for_cool,
)
from triagebox.rag import Bm25Index, chunk_text
from triagebox.serverproc import ServerProc

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
M15 = ROOT / "models" / "qwen2.5-1.5b-instruct"
M7 = ROOT / "models" / "qwen2.5-7b-instruct"
PROMPT_VERSION = 2


@dataclass(frozen=True)
class Config:
    name: str
    model: Path


CONFIGS = [
    Config("qwen2.5-1.5b-fp16", M15 / "qwen2.5-1.5b-instruct-fp16.gguf"),
    Config("qwen2.5-1.5b-q8_0", M15 / "qwen2.5-1.5b-instruct-q8_0.gguf"),
    Config("qwen2.5-1.5b-q4_k_m", M15 / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    Config("qwen2.5-7b-q8_0", M7 / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"),
    Config("qwen2.5-7b-q4_k_m", M7 / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
]


def server(cfg: Config) -> ServerProc:
    log = ROOT / "data" / "logs" / f"server_{cfg.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    return ServerProc(EXE, cfg.model, log=log)


def accuracy(args: argparse.Namespace) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUTF8": "1"}
    for cfg in CONFIGS:
        label = f"{cfg.name}-p{PROMPT_VERSION}"
        out = ROOT / "results" / f"{label}_{args.split}.json"
        if out.exists() and not args.force:
            print(f"skip {label}: {out.name} exists (use --force to redo)")
            continue
        print(f"=== {label} ===", flush=True)
        with server(cfg) as srv:
            cmd = [
                sys.executable, str(ROOT / "scripts" / "run_eval.py"),
                "--label", label, "--split", args.split, "--n-per-kind", str(args.n_per_kind),
                "--prompt-version", str(PROMPT_VERSION), "--server-url", srv.url, "--power",
            ]
            r = subprocess.run(cmd, env=env, cwd=ROOT)
            if r.returncode != 0:
                raise SystemExit(f"run_eval failed for {label}")


def med(xs):
    return statistics.median(xs) if xs else None


def burst(cfg: Config, rnd: int, index: Bm25Index, questions: list, args: argparse.Namespace) -> dict:
    with server(cfg) as srv:
        client = LlamaServer(srv.url, max_tokens=256, seed=0)
        llm = TimedLLM(client)
        for q in questions[:3]:  # warm-up: weights paged in, caches primed; discarded
            client.generate("Say OK.")
        cool = wait_for_cool(args.cool_c, timeout_s=args.cool_timeout)
        llm.timings.clear()
        with GpuPower() as power:
            t0 = time.perf_counter()
            for q in questions:
                triage(q.question, index, llm, k=4, prompt_version=PROMPT_VERSION)
            t1 = time.perf_counter()
            state = power.state(t0, t1)
            burst_j = energy_joules(power.samples, t0, t1)
            joules, watts = [], []
            for start, end, timing in llm.timings:
                w = power.mean_watts(start, end)
                if w is not None:
                    watts.append(w)
                    joules.append(w * timing.total_ms / 1000.0)
    steady = timing_steady(state)
    return {
        "config": cfg.name,
        "round": rnd,
        "start_temp_c": cool.temp_c,
        "cooled_to_target": cool.reached,
        "cooldown_s": round(cool.waited_s, 1),
        "n_requests": len(questions),
        "ttft_ms_median": med([t.ttft_ms for _, _, t in llm.timings]),
        "tokens_per_s_median": med([t.tokens_per_s for _, _, t in llm.timings if t.n_tokens > 1]),
        "tokens_median": med([t.n_tokens for _, _, t in llm.timings]),
        "gpu_watts_median": med(watts),
        "joules_per_answer_median": med(joules),  # per-request estimate; noisy for sub-100 ms answers
        "joules_per_answer_burst": None if burst_j is None else burst_j / len(questions),
        "burst_seconds": t1 - t0,
        "median_sm_mhz": state.median_sm_mhz,
        "clock_fraction": state.clock_fraction,
        "max_temp_c": state.max_temp_c,
        "throttle_mask": hex(state.throttle_reasons_seen),
        "power_capped": bool(state.throttle_reasons_seen & POWER_CAP_BIT),
        "hard_throttle": bool(state.throttle_reasons_seen & HARD_THROTTLE_BITS),
        "unconstrained_boost": timing_trustworthy(state),
        "trustworthy": steady and cool.reached,
    }


def timing(args: argparse.Namespace) -> None:
    docs = load_corpus(ROOT / "data" / "squad_v2" / "dev_corpus.jsonl")
    index = Bm25Index([c for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)])
    questions = [i for i in load_items(ROOT / "data" / "squad_v2" / "dev_questions.jsonl") if i.answerable][: args.burst]
    bursts: list[dict] = []
    out = ROOT / "results" / "timing_ladder.json"
    for rnd in range(args.rounds):
        order = CONFIGS[rnd % len(CONFIGS) :] + CONFIGS[: rnd % len(CONFIGS)]  # rotate so no config always goes first
        for cfg in order:
            b = burst(cfg, rnd, index, questions, args)
            bursts.append(b)
            print(
                f"round {rnd} {cfg.name:22s} start {b['start_temp_c']}C  clock {100 * (b['clock_fraction'] or 0):3.0f}%  "
                f"max {b['max_temp_c']}C  mask {b['throttle_mask']}  {b['tokens_per_s_median'] or 0:6.1f} tok/s  "
                f"{'ACCEPTED' if b['trustworthy'] else 'rejected (hard throttle, too hot, or not cooled)'}",
                flush=True,
            )
            out.write_text(json.dumps({"bursts": bursts, "summary": summarize(bursts), "meta": meta(args)}, indent=2), encoding="utf-8")
    print()
    print_summary(summarize(bursts))


def summarize(bursts: list[dict]) -> dict:
    res = {}
    for name in dict.fromkeys(b["config"] for b in bursts):
        mine = [b for b in bursts if b["config"] == name]
        ok = [b for b in mine if b["trustworthy"]]
        def stat(key):
            vals = [b[key] for b in ok if b[key] is not None]
            return None if not vals else {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        res[name] = {
            "bursts": len(mine),
            "accepted": len(ok),
            "tokens_per_s": stat("tokens_per_s_median"),
            "ttft_ms": stat("ttft_ms_median"),
            "joules_per_answer": stat("joules_per_answer_burst"),
            "tokens_per_s_spread": (max(b["tokens_per_s_median"] for b in ok) / min(b["tokens_per_s_median"] for b in ok)) if ok else None,
            "power_capped_bursts": sum(b["power_capped"] for b in ok),
            "median_clock_fraction": statistics.median(b["clock_fraction"] for b in ok) if ok else None,
            "gpu_watts": stat("gpu_watts_median"),
        }
    return res


def print_summary(summary: dict) -> None:
    print(f"{'configuration':22s} {'bursts ok':>9s} {'tok/s (median, range)':>26s} {'ttft ms':>9s} {'J/answer':>9s}")
    for name, s in summary.items():
        if s["tokens_per_s"] is None:
            print(f"{name:22s} {s['accepted']:>4d}/{s['bursts']:<4d} {'no trustworthy burst':>26s}")
            continue
        t, f, j = s["tokens_per_s"], s["ttft_ms"], s["joules_per_answer"]
        print(
            f"{name:22s} {s['accepted']:>4d}/{s['bursts']:<4d} {t['median']:8.1f} ({t['min']:.0f}-{t['max']:.0f})        "
            f"{f['median']:9.0f} {j['median'] if j else float('nan'):9.1f}"
        )


def meta(args: argparse.Namespace) -> dict:
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rounds": args.rounds,
        "burst_requests": args.burst,
        "cool_target_c": args.cool_c,
        "prompt_version": PROMPT_VERSION,
        "rule": "a burst counts only if the GPU cooled to the target before it, reported no thermal, hardware-slowdown or "
        "power-brake throttle reason, and stayed at or below 85 C. A software power cap (operation at the board's rated "
        "power limit) is tolerated and recorded, so numbers describe this board at its power limit, not unconstrained boost. "
        "Energy is GPU board power integrated over the whole burst divided by the number of answers (includes small gaps between requests); it is GPU board power, not system power.",
        "llama_cpp": "b11700, CUDA 12.4, all layers on GPU, context 4096, temperature 0",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accuracy")
    a.add_argument("--split", choices=["dev", "test"], default="dev")
    a.add_argument("--n-per-kind", type=int, default=100)
    a.add_argument("--force", action="store_true")
    t = sub.add_parser("timing")
    t.add_argument("--rounds", type=int, default=3)
    t.add_argument("--burst", type=int, default=8)
    t.add_argument("--cool-c", type=int, default=60)
    t.add_argument("--cool-timeout", type=float, default=900.0)
    args = ap.parse_args()
    {"accuracy": accuracy, "timing": timing}[args.cmd](args)


if __name__ == "__main__":
    main()
