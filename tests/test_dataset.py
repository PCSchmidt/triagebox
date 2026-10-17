from triagebox.dataset import load_corpus, load_items, save_corpus, save_items, squad_to_dataset


def tiny_squad():
    def para(prefix, n_ans, n_imp):
        qas = [
            {"id": f"{prefix}-a{i}", "question": f"{prefix} q{i}?", "is_impossible": False,
             "answers": [{"text": f"ans{i}"}, {"text": f"ans{i}"}, {"text": f"alt{i}"}]}
            for i in range(n_ans)
        ] + [
            {"id": f"{prefix}-u{i}", "question": f"{prefix} trick{i}?", "is_impossible": True, "answers": []}
            for i in range(n_imp)
        ]
        return {"context": f"context text for {prefix}", "qas": qas}

    return {
        "data": [
            {"title": "A", "paragraphs": [para("A0", 3, 2), para("A1", 2, 2)]},
            {"title": "B", "paragraphs": [para("B0", 4, 3)]},
            {"title": "C", "paragraphs": [para("C0", 2, 2)]},
        ]
    }


def test_converter_counts_and_gold_fields():
    docs, items = squad_to_dataset(tiny_squad(), n_articles=3, n_answerable=6, n_unanswerable=5, seed=1)
    assert len(items) == 11
    assert sum(i.answerable for i in items) == 6 and sum(not i.answerable for i in items) == 5
    for it in items:
        if it.answerable:
            assert it.gold_docs and it.gold_docs[0] in docs
            assert it.gold_answer.startswith("ans") and it.alt_answers == (f"alt{it.gold_answer[3:]}",)
        else:
            assert it.gold_docs == () and it.gold_answer is None


def test_converter_is_deterministic_and_seed_sensitive():
    a = squad_to_dataset(tiny_squad(), 2, 4, 3, seed=7)
    b = squad_to_dataset(tiny_squad(), 2, 4, 3, seed=7)
    assert a == b
    assert squad_to_dataset(tiny_squad(), 2, 4, 3, seed=8) != a


def test_every_paragraph_of_a_chosen_article_is_a_document():
    docs, _ = squad_to_dataset(tiny_squad(), n_articles=1, n_answerable=1, n_unanswerable=1, seed=0)
    prefixes = {text.split()[-1][0] for text in docs.values()}
    assert len(prefixes) == 1  # all documents come from one article
    assert len(docs) in (1, 2)


def test_requests_larger_than_available_are_capped():
    _, items = squad_to_dataset(tiny_squad(), 3, 999, 999, seed=0)
    assert len(items) == (3 + 2 + 4 + 2) + (2 + 2 + 3 + 2)


def test_jsonl_round_trips(tmp_path):
    docs, items = squad_to_dataset(tiny_squad(), 3, 4, 4, seed=0)
    save_items(tmp_path / "q.jsonl", items)
    save_corpus(tmp_path / "c.jsonl", docs)
    assert load_items(tmp_path / "q.jsonl") == items
    assert load_corpus(tmp_path / "c.jsonl") == docs


def test_explicit_article_ids_select_exactly_those_articles():
    docs, items = squad_to_dataset(tiny_squad(), 0, 999, 999, seed=0, article_ids=[1])
    assert all(d.startswith("a1") for d in docs)
    assert all(i.gold_docs[0].startswith("a1") for i in items if i.answerable)
