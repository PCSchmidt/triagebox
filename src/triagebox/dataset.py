"""Evaluation data: corpus documents and questions, as JSON Lines, plus a SQuAD 2.0 converter.

SQuAD 2.0 (Rajpurkar, Jia and Liang, 2018; CC BY-SA 4.0) pairs Wikipedia paragraphs
with questions. About half its questions are *unanswerable*: written by crowdworkers
to look answerable from a paragraph that does not contain the answer. That makes
them near-miss unanswerables, a hard test of whether a model refuses or invents.
Here each paragraph becomes one document.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .metrics import EvalItem


def save_items(path: Path, items: list[EvalItem]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), sort_keys=True) + "\n")


def load_items(path: Path) -> list[EvalItem]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        d["gold_docs"] = tuple(d["gold_docs"])
        d["alt_answers"] = tuple(d.get("alt_answers", ()))
        out.append(EvalItem(**d))
    return out


def save_corpus(path: Path, docs: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc_id, text in docs.items():
            f.write(json.dumps({"doc_id": doc_id, "text": text}) + "\n")


def load_corpus(path: Path) -> dict[str, str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {r["doc_id"]: r["text"] for r in rows}


def squad_to_dataset(
    squad: dict,
    n_articles: int,
    n_answerable: int,
    n_unanswerable: int,
    seed: int = 0,
    article_ids: list[int] | None = None,
) -> tuple[dict[str, str], list[EvalItem]]:
    """Convert a SQuAD 2.0 file (already parsed) to (corpus, questions).

    Picks ``n_articles`` articles at random, uses every paragraph of those articles
    as a document (so retrieval has distractors), and samples the requested number
    of answerable and unanswerable questions from them. Deterministic for a seed.
    ``article_ids`` (indices into ``squad["data"]``) overrides the random choice of articles.
    """
    rng = random.Random(seed)
    articles = list(enumerate(squad["data"]))
    if article_ids is not None:
        chosen = [articles[i] for i in sorted(article_ids)]
    else:
        chosen = sorted(rng.sample(articles, min(n_articles, len(articles))))

    docs: dict[str, str] = {}
    answerable: list[EvalItem] = []
    unanswerable: list[EvalItem] = []
    for ai, article in chosen:
        for pi, para in enumerate(article["paragraphs"]):
            doc_id = f"a{ai}p{pi}"
            docs[doc_id] = para["context"]
            for qa in para["qas"]:
                if qa.get("is_impossible"):
                    unanswerable.append(EvalItem(qa["id"], qa["question"], False))
                else:
                    texts = list(dict.fromkeys(a["text"] for a in qa["answers"]))
                    answerable.append(
                        EvalItem(qa["id"], qa["question"], True, (doc_id,), texts[0], tuple(texts[1:]))
                    )
    items = rng.sample(answerable, min(n_answerable, len(answerable))) + rng.sample(
        unanswerable, min(n_unanswerable, len(unanswerable))
    )
    return docs, items
