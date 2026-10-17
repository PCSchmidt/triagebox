"""Draw the quality-versus-energy chart for the quantisation ladder as a static SVG.

    python scripts/make_ladder_chart.py   ->   docs/ladder-quality-vs-energy.svg

Every number comes from the saved result files (``results/<config>-p2_dev.json``
for accuracy and hallucination, ``results/timing_ladder.json`` for energy), so the
chart cannot drift from the tables. Two panels share the energy axis: accuracy
(higher is better) and hallucination (lower is better). Series colour is model
size (validated categorical slots 1 and 2); marker shape repeats it (circle = 1.5B,
square = 7B) so identity never rests on colour alone; precision is direct-labelled.
Light and dark colours are selected, not inverted. The efficient frontier (no other
configuration is both more accurate and cheaper) is drawn on the accuracy panel.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "ladder-quality-vs-energy.svg"

CONFIGS = [
    # name, size, precision, label placement per panel: (anchor, dx, dy)
    ("qwen2.5-1.5b-fp16", "1.5B", "FP16", {"acc": ("middle", 0, 24), "hal": ("start", 11, 4)}),
    ("qwen2.5-1.5b-q8_0", "1.5B", "Q8_0", {"acc": ("middle", 0, -13), "hal": ("end", -11, 4)}),
    ("qwen2.5-1.5b-q4_k_m", "1.5B", "Q4_K_M", {"acc": ("start", 11, 4), "hal": ("start", 11, 4)}),
    ("qwen2.5-7b-q8_0", "7B", "Q8_0", {"acc": ("middle", 0, -13), "hal": ("middle", 0, -13)}),
    ("qwen2.5-7b-q4_k_m", "7B", "Q4_K_M", {"acc": ("middle", 0, -13), "hal": ("middle", 0, 24)}),
]

W, H = 920, 480
PANEL_W, PLOT_TOP, PLOT_BOTTOM = 370, 120, 370
PANELS = {"acc": 60, "hal": 530}  # left edge of each plot area
XMAX = 70.0


def load_points() -> list[dict]:
    timing = json.loads((ROOT / "results" / "timing_ladder.json").read_text(encoding="utf-8"))["summary"]
    pts = []
    for name, size, prec, place in CONFIGS:
        res = json.loads((ROOT / "results" / f"{name}-p2_dev.json").read_text(encoding="utf-8"))["system_level"]
        pts.append(
            {
                "name": name,
                "size": size,
                "prec": prec,
                "joules": timing[name]["joules_per_answer"]["median"],
                "acc": 100 * res["accuracy"],
                "hal": 100 * res["hallucination_rate"],
                "place": place,
            }
        )
    return pts


def frontier(points: list[dict]) -> list[dict]:
    """Points not beaten on both accuracy (higher) and energy (lower), cheapest first."""
    keep = []
    for p in points:
        beaten = any(
            q is not p
            and q["acc"] >= p["acc"]
            and q["joules"] <= p["joules"]
            and (q["acc"] > p["acc"] or q["joules"] < p["joules"])
            for q in points
        )
        if not beaten:
            keep.append(p)
    return sorted(keep, key=lambda p: p["joules"])


def main() -> None:
    pts = load_points()
    front = frontier(pts)

    def px(panel: str, joules: float) -> float:
        return PANELS[panel] + joules / XMAX * PANEL_W

    def py(value: float) -> float:
        return PLOT_BOTTOM - value / 100.0 * (PLOT_BOTTOM - PLOT_TOP)

    o: list[str] = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-labelledby="t d">')
    add("<title id=\"t\">Quality versus GPU energy per answer across the quantisation ladder</title>")
    add("<desc id=\"d\">Two scatter panels over GPU energy per answer. Left: accuracy. Right: hallucination rate on "
        "unanswerable questions. Five configurations of Qwen2.5-Instruct: 1.5B at FP16, Q8_0 and Q4_K_M, and 7B at "
        "Q8_0 and Q4_K_M. The 7B models are far more accurate and far less prone to hallucination but cost about "
        "four times the energy per answer. Values are in the accompanying table.</desc>")
    add("""<style>
:root{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834}
@media (prefers-color-scheme: dark){:root{--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926}}
text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif}
.bg{fill:var(--surface)} .h1{font-size:16px;font-weight:600;fill:var(--ink)} .h2{font-size:12px;fill:var(--ink2)}
.panel{font-size:13px;font-weight:600;fill:var(--ink)} .tick{font-size:11px;fill:var(--muted);font-variant-numeric:tabular-nums}
.axis-title{font-size:12px;fill:var(--ink2)} .lab{font-size:12px;fill:var(--ink2)} .note{font-size:11px;fill:var(--muted)}
.grid{stroke:var(--grid);stroke-width:1} .base{stroke:var(--axis);stroke-width:1} .front{stroke:var(--muted);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
.m1{fill:var(--s1)} .m2{fill:var(--s2)} .ring{stroke:var(--surface);stroke-width:2}
</style>""")
    add(f'<rect class="bg" width="{W}" height="{H}"/>')
    add('<text class="h1" x="24" y="30">Quality versus GPU energy per answer</text>')
    add('<text class="h2" x="24" y="50">Qwen2.5-Instruct on the SQuAD 2.0 dev subset (100 + 100 questions), after the citation policy. '
        'RTX A4500 laptop GPU at its power limit.</text>')
    # legend (two series, so always present)
    add('<circle class="m1 ring" cx="32" cy="80" r="6"/><text class="lab" x="44" y="84">1.5B parameters</text>')
    add('<rect class="m2 ring" x="166" y="74" width="12" height="12" rx="2"/><text class="lab" x="184" y="84">7B parameters</text>')
    add('<text class="note" x="290" y="84">Precision is labelled on each point. A line joins the efficient set.</text>')

    panels = {"acc": "Accuracy (higher is better)", "hal": "Hallucination on unanswerable questions (lower is better)"}
    for key, title in panels.items():
        x0 = PANELS[key]
        add(f'<text class="panel" x="{x0}" y="{PLOT_TOP - 14}">{escape(title)}</text>')
        for v in range(0, 101, 20):
            y = py(v)
            add(f'<line class="{"base" if v == 0 else "grid"}" x1="{x0}" x2="{x0 + PANEL_W}" y1="{y:.1f}" y2="{y:.1f}"/>')
            add(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{v}%</text>')
        for j in range(0, int(XMAX) + 1, 10):
            x = px(key, j)
            add(f'<text class="tick" x="{x:.1f}" y="{PLOT_BOTTOM + 18}" text-anchor="middle">{j}</text>')
        add(f'<text class="axis-title" x="{x0 + PANEL_W / 2}" y="{PLOT_BOTTOM + 40}" text-anchor="middle">'
            f'GPU energy per answer (joules, lower is better)</text>')
        if key == "acc":
            path = " ".join(f"{px(key, p['joules']):.1f},{py(p[key]):.1f}" for p in front)
            add(f'<polyline class="front" points="{path}"/>')
        for p in pts:
            cx, cy = px(key, p["joules"]), py(p[key])
            cls = "m1" if p["size"] == "1.5B" else "m2"
            tip = (f'{p["size"]} {p["prec"]}: {p["joules"]:.1f} J per answer, accuracy {p["acc"]:.0f}%, '
                   f'hallucination {p["hal"]:.0f}%')
            if p["size"] == "1.5B":
                add(f'<circle class="{cls} ring" cx="{cx:.1f}" cy="{cy:.1f}" r="6"><title>{escape(tip)}</title></circle>')
            else:
                add(f'<rect class="{cls} ring" x="{cx - 6:.1f}" y="{cy - 6:.1f}" width="12" height="12" rx="2">'
                    f'<title>{escape(tip)}</title></rect>')
            anchor, dx, dy = p["place"][key]
            add(f'<text class="lab" x="{cx + dx:.1f}" y="{cy + dy:.1f}" text-anchor="{anchor}">{escape(p["prec"])}</text>')

    add(f'<text class="note" x="24" y="{H - 22}">Accuracy is capped at 91% by retrieval (gold paragraph in the top 4). One run per '
        f'configuration; 100 questions per kind, so differences of a few points are noise.</text>')
    add(f'<text class="note" x="24" y="{H - 8}">GPU board energy (not system power), integrated over cooled, short bursts. '
        f'Not an edge-device measurement. Values: docs/ladder.md.</text>')
    add("</svg>")
    OUT.write_text("\n".join(o) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print("efficient set (accuracy vs energy):", [(p["size"], p["prec"], round(p["joules"], 1), round(p["acc"])) for p in front])


if __name__ == "__main__":
    main()
