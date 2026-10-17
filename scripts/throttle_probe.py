"""Does speed drift during sustained generation, and does the GPU say why?

    python scripts/throttle_probe.py --requests 120

Sends the same retrieval-augmented request repeatedly to a running llama-server
while logging GPU clocks, temperature, power and throttle reasons, then prints
per-block speed beside the GPU state. A drop in clock or power with a throttle
reason set means speed measured late in a run is not comparable to early speed.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import threading
import time
from pathlib import Path

from triagebox.dataset import load_corpus, load_items
from triagebox.llamacpp import LlamaServer, TimedLLM
from triagebox.pipeline import triage
from triagebox.rag import Bm25Index, chunk_text

DATA = Path(__file__).resolve().parents[1] / "data" / "squad_v2"
FIELDS = "clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active"
NAMES = {
    0x01: "idle",
    0x02: "app-clocks",
    0x04: "sw-power-cap",
    0x08: "hw-slowdown",
    0x20: "sw-thermal",
    0x40: "hw-thermal",
    0x80: "hw-power-brake",
}


def reasons(mask: int) -> str:
    return ",".join(n for bit, n in NAMES.items() if mask & bit) or "none"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=120)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    args = ap.parse_args()

    docs = load_corpus(DATA / "dev_corpus.jsonl")
    item = next(i for i in load_items(DATA / "dev_questions.jsonl") if i.answerable)
    index = Bm25Index([c for d, t in docs.items() for c in chunk_text(d, t, max_words=200, overlap=0)])

    log: list[tuple[float, int, int, float, int]] = []
    stop = threading.Event()

    def sample() -> None:
        proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits", "-lms=500"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        for line in proc.stdout:
            if stop.is_set():
                break
            try:
                sm, temp, watts, mask = [x.strip() for x in line.split(",")]
                log.append((time.perf_counter(), int(sm), int(temp), float(watts), int(mask, 16)))
            except ValueError:
                pass
        proc.terminate()

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    time.sleep(1.0)

    llm = TimedLLM(LlamaServer(args.url, max_tokens=64, seed=0))
    print(f"{'requests':>10} {'ttft ms':>8} {'tok/s':>6} | {'SM MHz':>6} {'temp C':>6} {'W':>5}  throttle reasons seen")
    for n in range(args.requests):
        triage(item.question, index, llm, k=4)
        if (n + 1) % args.block == 0:
            block = llm.timings[-args.block :]
            t0, t1 = block[0][0], block[-1][1]
            g = [s for s in log if t0 <= s[0] <= t1]
            seen = 0
            for s in g:
                seen |= s[4]
            ttft = statistics.median(b[2].ttft_ms for b in block)
            tps = statistics.median(b[2].tokens_per_s for b in block if b[2].n_tokens > 1)
            if g:
                print(f"{n - args.block + 2:>4}-{n + 1:<5} {ttft:8.0f} {tps:6.1f} | {statistics.median(s[1] for s in g):6.0f} "
                      f"{max(s[2] for s in g):6d} {statistics.median(s[3] for s in g):5.0f}  {reasons(seen)}")
    stop.set()


if __name__ == "__main__":
    main()
