from triagebox.verify import Support, clean_quote, question_entities, verify

PASSAGE = "Warsaw stands on the Vistula River in Poland. About 15 kilometres from Warsaw, the Vistula river's environment changes strikingly."


def test_none_and_empty_replies_give_no_quote():
    assert clean_quote("NONE") is None and clean_quote(" none. ") is None and clean_quote("") is None


def test_quote_cleanup_strips_quotes_and_citation_ids():
    assert clean_quote('"Warsaw stands on the Vistula River in Poland." [a1#0]') == "Warsaw stands on the Vistula River in Poland."


def test_verbatim_quote_covering_question_and_answer_scores_high():
    s = verify("Where does Warsaw stand?", "on the Vistula River", "Warsaw stands on the Vistula River in Poland.", [PASSAGE])
    assert s.found and s.answer_coverage == 1.0 and s.entity_coverage == 1.0


def test_invented_quote_is_not_found_whatever_it_says():
    s = verify("Where does Warsaw stand?", "on the Vistula River", "Warsaw stands on the Danube.", [PASSAGE])
    assert not s.found


def test_right_answer_but_wrong_entity_in_the_question_scores_low():
    # the passage says 15 km from *Warsaw*; the question asks about Czerniakow
    q = "How far from Czerniakow does the Vistula river's environment change noticeably?"
    s = verify(q, "15 kilometres", "About 15 kilometres from Warsaw, the Vistula river's environment changes strikingly.", [PASSAGE])
    assert s.found and s.answer_coverage == 1.0 and s.entity_coverage <= 0.5  # 'Vistula' is there, 'Czerniakow' is not


def test_missing_quote_scores_zero():
    s = verify("Where is Warsaw?", "a", None, [PASSAGE])
    assert s.match == 0.0 and s.answer_coverage == 0.0 and not s.found
    assert s.entity_coverage == 1.0  # does not depend on the model producing a quote


def test_entities_are_names_and_numbers_not_the_first_word_or_question_words():
    assert question_entities("Which ship did Karelia Sun pass at 0441Z near Cape Orlin?") >= {"karelia", "sun", "0441z", "cape", "orlin"}
    assert "which" not in question_entities("Which ship was it?") and "ship" not in question_entities("Which ship was it?")


def test_accents_are_folded_in_entities():
    assert verify("Near Czerniakow?", "x", "Czerniaków Lake is nearby.", ["Czerniaków Lake is nearby."]).entity_coverage == 1.0


def test_near_verbatim_quote_still_counts_as_found():
    q = "About 15 kilometres from Warsaw, the Vistula river's environment changes strikingly"
    assert verify("q?", "15 kilometres", q, [PASSAGE]).found
