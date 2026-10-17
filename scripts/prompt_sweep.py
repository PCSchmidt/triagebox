"""Accuracy-only runs of several prompt versions over several models (dev split).

    python scripts/prompt_sweep.py --models 1.5b-q8_0 1.5b-q4_k_m --prompts 3 4

One llama-server per model, then run_eval.py once per prompt version. No timing or
energy is collected (see ladder.py for that). Labels follow <model>-p<version>.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from triagebox.serverproc import ServerProc

ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "tools" / "llama-b11700" / "llama-server.exe"
M15 = ROOT / "models" / "qwen2.5-1.5b-instruct"
M7 = ROOT / "models" / "qwen2.5-7b-instruct"
MODELS = {
    "1.5b-fp16": ("qwen2.5-1.5b-fp16", M15 / "qwen2.5-1.5b-instruct-fp16.gguf"),
    "1.5b-q8_0": ("qwen2.5-1.5b-q8_0", M15 / "qwen2.5-1.5b-instruct-q8_0.gguf"),
    "1.5b-q4_k_m": ("qwen2.5-1.5b-q4_k_m", M15 / "qwen2.5-1.5b-instruct-q4_k_m.gguf"),
    "7b-q8_0": ("qwen2.5-7b-q8_0", M7 / "qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf"),
    "7b-q4_k_m": ("qwen2.5-7b-q4_k_m", M7 / "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), required=True)
    ap.add_argument("--prompts", nargs="+", type=int, required=True)
    ap.add_argument("--dataset", choices=["squad", "isr"], default="squad")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()
    for key in args.models:
        label, path = MODELS[key]
        with ServerProc(EXE, path, log=ROOT / "data" / "logs" / f"sweep_{key}.log") as s:
            for p in args.prompts:
                print(f"== {label} prompt v{p}", flush=True)
                subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "run_eval.py"), "--server-url", s.url,
                     "--label", (f"isr-{label}-p{p}" if args.dataset == "isr" else f"{label}-p{p}"), "--dataset", args.dataset, "--split", args.split, "--n-per-kind", "100", "--prompt-version", str(p)],
                    check=True,
                )


if __name__ == "__main__":
    main()
