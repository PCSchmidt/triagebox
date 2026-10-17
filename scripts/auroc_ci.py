"""Bootstrap 95% intervals for detector AUROCs on a labelled set (data/detector/<source>_<split>.jsonl).

    python scripts/auroc_ci.py squad dev
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from detector_eval import auroc  # noqa: E402

rows = [json.loads(l) for l in (ROOT / "data" / "detector" / f"{sys.argv[1]}_{sys.argv[2]}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
for r in rows:
    r["s_found"] = float(r["found"])
    r["s_found_min"] = r["s_found"] * min(r["a_cov"], r["e_cov"])
KEYS = {"lexical_overlap": "lexical", "entity_coverage": "e_cov", "quote_found": "s_found", "found_x_min(answer,entity)": "s_found_min"}
pos = [r for r in rows if r["label"] == 1]
neg = [r for r in rows if r["label"] == 0]
rng = random.Random(0)
print(f"{sys.argv[1]} {sys.argv[2]}: {len(pos)} supported, {len(neg)} unsupported")
samples = []
for _ in range(1000):
    samples.append([rng.choice(pos) for _ in pos] + [rng.choice(neg) for _ in neg])
for name, key in KEYS.items():
    est = auroc([r[key] for r in rows], [r["label"] for r in rows])
    boots = sorted(auroc([r[key] for r in smp], [r["label"] for r in smp]) for smp in samples)
    print(f"  {name:30s} {est:.3f}  [{boots[25]:.2f}-{boots[974]:.2f}]")
# paired difference: entity coverage minus lexical overlap
diffs = sorted(auroc([r["e_cov"] for r in smp], [r["label"] for r in smp]) - auroc([r["lexical"] for r in smp], [r["label"] for r in smp]) for smp in samples)
print(f"  entity - lexical AUROC difference, paired: [{diffs[25]:.2f}, {diffs[974]:.2f}]")
