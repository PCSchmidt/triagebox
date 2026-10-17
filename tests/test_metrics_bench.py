import pytest

from triagebox.bench import GenTiming, Point, joules_per_answer, measure_generation, pareto_front
from triagebox.metrics import EvalItem, Outcome, score
from triagebox.prompt import ParsedResponse


def answer(text, *cites):
    return ParsedResponse("answer", text, tuple(cites))


REFUSAL = ParsedResponse("refusal", "")
MALFORMED = ParsedResponse("malformed", "???")


def outcome(item, parsed, retrieved=("d1#0", "d2#0")):
    return Outcome(item, parsed, frozenset(retrieved))


def items():
    return {
        "a1": EvalItem("a1", "q", True, ("d1",), "three trucks"),
        "a2": EvalItem("a2", "q", True, ("d2",), "northern port"),
        "u1": EvalItem("u1", "q", False),
        "u2": EvalItem("u2", "q", False),
    }


def test_scores_on_a_mixed_set():
    it = items()
    s = score(
        [
            outcome(it["a1"], answer("There were Three  Trucks [d1#0].", "d1#0")),  # correct, gold cited
            outcome(it["a2"], REFUSAL),  # false refusal
            outcome(it["u1"], REFUSAL),  # correct refusal
            outcome(it["u2"], answer("Invented claim [zz#1].", "zz#1")),  # hallucination, invalid citation
        ]
    )
    assert (s.n_answerable, s.n_unanswerable) == (2, 2)
    assert s.accuracy == 0.5
    assert s.false_refusal_rate == 0.5
    assert s.hallucination_rate == 0.5
    assert s.citation_validity == 0.5  # 1 valid of 2 cited
    assert s.gold_citation_recall == 1.0  # the one answered answerable item cited a gold doc
    assert s.malformed_rate == 0.0


def test_malformed_and_missing_denominators():
    it = items()
    s = score([outcome(it["u1"], MALFORMED)])
    assert s.malformed_rate == 1.0
    assert s.accuracy is None and s.false_refusal_rate is None  # no answerable items
    assert s.hallucination_rate == 0.0  # malformed is not an answer
    assert s.citation_validity is None  # nothing cited


def test_wrong_document_citation_lowers_gold_recall():
    it = items()
    s = score([outcome(it["a1"], answer("three trucks [d2#0]", "d2#0"))])
    assert s.accuracy == 1.0 and s.gold_citation_recall == 0.0


def test_measure_generation_with_a_fake_clock():
    times = iter([0.0, 0.5, 1.0])  # the timer reads the clock at start, at the first token, and at the end

    def clock():
        return next(times)

    t = measure_generation(iter(["a", "b", "c", "d"]), clock=clock)
    assert t.ttft_ms == pytest.approx(500.0)
    assert t.n_tokens == 4
    # 3 tokens after the first, decoded between t=0.5 and the end (t=1.0)
    assert t.tokens_per_s == pytest.approx(3 / 0.5)
    assert t.total_ms == pytest.approx(1000.0)


def test_measure_generation_needs_tokens_and_single_token_rate_is_zero():
    with pytest.raises(ValueError):
        measure_generation(iter([]))
    t = measure_generation(iter(["x"]), clock=iter([0.0, 0.1, 0.2]).__next__)
    assert t.n_tokens == 1 and t.tokens_per_s == 0.0


def test_joules_per_answer():
    assert joules_per_answer(20.0, 2500.0) == pytest.approx(50.0)


def test_pareto_front_drops_dominated_points():
    pts = [
        Point("fp16", 0.80, 100.0),
        Point("int8", 0.78, 60.0),
        Point("int4", 0.70, 40.0),
        Point("bad", 0.60, 90.0),  # dominated by int8 and int4
        Point("dup-ish", 0.78, 70.0),  # dominated by int8
    ]
    front = pareto_front(pts)
    assert [p.name for p in front] == ["int4", "int8", "fp16"]


def test_equal_points_both_stay_on_the_front():
    a, b = Point("a", 0.5, 10.0), Point("b", 0.5, 10.0)
    assert len(pareto_front([a, b])) == 2


def test_any_alternative_gold_answer_counts():
    item = EvalItem("a", "q", True, ("d1",), "Warsaw", alt_answers=("the Polish capital",))
    s = score([outcome(item, answer("It is the Polish capital [d1#0].", "d1#0"))])
    assert s.accuracy == 1.0
    s = score([outcome(item, answer("It is Berlin [d1#0].", "d1#0"))])
    assert s.accuracy == 0.0
