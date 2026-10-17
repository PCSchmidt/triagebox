"""Print the ladder tables (markdown) from the saved result files.

    python scripts/ladder_table.py

Reads ``results/<config>-p2_dev.json`` for accuracy and, if present,
``results/timing_ladder.json`` for speed and energy.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDER = [
    ("qwen2.5-1.5b-fp16", "1.5B", "FP16"),
    ("qwen2.5-1.5b-q8_0", "1.5B", "Q8_0"),
    ("qwen2.5-1.5b-q4_k_m", "1.5B", "Q4_K_M"),
    ("qwen2.5-7b-q8_0", "7B", "Q8_0"),
    ("qwen2.5-7b-q4_k_m", "7B", "Q4_K_M"),
]


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def load(name):
    p = ROOT / "results" / f"{name}-p2_dev.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def accuracy_tables() -> str:
    head = "| Model | Precision | Accuracy | Hallucination | False refusal | Citation validity | Malformed |\n| --- | --- | --- | --- | --- | --- | --- |"
    model, system = [head], [head]
    for name, size, prec in ORDER:
        r = load(name)
        if not r:
            continue
        for lines, level in ((model, "model_level"), (system, "system_level")):
            s = r[level]
            lines.append(
                f"| {size} | {prec} | {pct(s['accuracy'])} | {pct(s['hallucination_rate'])} | {pct(s['false_refusal_rate'])} | "
                f"{pct(s['citation_validity'])} | {pct(s['malformed_rate'])} |"
            )
    return "**Model level (what the model said)**\n\n" + "\n".join(model) + "\n\n**System level (after the citation policy)**\n\n" + "\n".join(system)


def timing_table() -> str | None:
    p = ROOT / "results" / "timing_ladder.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    bursts = data["bursts"]
    lines = [
        "| Model | Precision | Bursts accepted | Decode tokens/s, median (range) | Round-to-round spread | Time to first token (ms) | GPU energy per answer (J), median (per round) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, size, prec in ORDER:
        s = data["summary"].get(name)
        if not s:
            continue
        acc = f"{s['accepted']}/{s['bursts']}"
        if s["tokens_per_s"] is None:
            lines.append(f"| {size} | {prec} | {acc} | no accepted burst | | | |")
            continue
        t, f, j = s["tokens_per_s"], s["ttft_ms"], s["joules_per_answer"]
        per_round = ", ".join(f"{b['joules_per_answer_burst']:.1f}" for b in bursts if b["config"] == name and b["trustworthy"])
        lines.append(
            f"| {size} | {prec} | {acc} | {t['median']:.1f} ({t['min']:.1f}-{t['max']:.1f}) | x{s['tokens_per_s_spread']:.2f} | "
            f"{f['median']:.0f} | {j['median']:.1f} ({per_round}) |"
        )
    masks = sorted({b["throttle_mask"] for b in bursts})
    hot = max(b["max_temp_c"] for b in bursts)
    lines.append("")
    lines.append(f"Throttle masks seen: {', '.join(masks)}; hard throttling in any burst: {any(b['hard_throttle'] for b in bursts)}; hottest burst: {hot} C.")
    return "\n".join(lines)


def main() -> None:
    print(accuracy_tables())
    t = timing_table()
    if t:
        print("\n**Speed and energy (accepted bursts only)**\n\n" + t)


if __name__ == "__main__":
    main()
