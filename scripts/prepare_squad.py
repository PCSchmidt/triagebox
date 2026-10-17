"""Download SQuAD 2.0 (dev) and build disjoint dev and test splits by article.

    python scripts/prepare_squad.py

SQuAD 2.0: Rajpurkar, Jia and Liang (2018), CC BY-SA 4.0,
https://rajpurkar.github.io/SQuAD-explorer/. The 35 articles of its dev file are
shuffled with a fixed seed; 18 become our development split and 17 our test split,
so no article (and no paragraph) appears in both. The test split is for final
numbers only; do not tune prompts or retrieval on it.
"""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

from triagebox.dataset import save_corpus, save_items, squad_to_dataset

URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
OUT = Path(__file__).resolve().parents[1] / "data" / "squad_v2"
N_PER_KIND = 300
SEED = 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "dev-v2.0.json"
    if not raw_path.exists():
        print("downloading", URL)
        raw_path.write_bytes(urllib.request.urlopen(URL, timeout=60).read())
    raw = raw_path.read_bytes()
    squad = json.loads(raw)

    order = list(range(len(squad["data"])))
    random.Random(SEED).shuffle(order)
    split = {"dev": sorted(order[:18]), "test": sorted(order[18:])}

    lines = [f"source: {URL}", f"sha256: {hashlib.sha256(raw).hexdigest()}", f"articles: {len(order)}", f"seed: {SEED}"]
    for name, ids in split.items():
        docs, items = squad_to_dataset(squad, 0, N_PER_KIND, N_PER_KIND, seed=SEED, article_ids=ids)
        save_corpus(OUT / f"{name}_corpus.jsonl", docs)
        save_items(OUT / f"{name}_questions.jsonl", items)
        n_a = sum(i.answerable for i in items)
        lines.append(f"{name}: articles {ids} -> {len(docs)} documents, {n_a} answerable + {len(items) - n_a} unanswerable questions")
        print(lines[-1])
    (OUT / "SOURCE.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
