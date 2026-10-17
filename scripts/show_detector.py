"""Print the headline table from results/detector_<source>_<split>.json."""
import json
import sys
from pathlib import Path

d = json.loads((Path(__file__).resolve().parents[1] / "results" / f"detector_{sys.argv[1]}_{sys.argv[2]}.json").read_text(encoding="utf-8"))
print(f"{d['source']} {d['split']}: n={d['n']} supported={d['supported']} unsupported={d['unsupported']}")
for k, v in d["auroc"].items():
    print(f"  AUROC {k:32s}{'n/a' if v is None else format(v, '.3f')}")
for k, pts in d.items():
    if k.startswith("operating_points"):
        print(" ", k)
        for p in pts:
            print(f"    t>={p['threshold']}: keeps {100 * p['supported_kept']:.0f}% of supported, removes {100 * p['unsupported_removed']:.0f}% of unsupported")
