from triagebox.judge import parse_verdict


def test_supported_and_not_supported_are_distinguished():
    assert parse_verdict("SUPPORTED - the passage says so") == "SUPPORTED"
    assert parse_verdict("NOT_SUPPORTED - it is about something else") == "NOT_SUPPORTED"
    assert parse_verdict("NOT_SUPPORTED-no dash space") == "NOT_SUPPORTED"


def test_case_and_leading_space_are_tolerated():
    assert parse_verdict("  supported - ok") == "SUPPORTED"


def test_anything_else_is_unparsed_not_guessed():
    assert parse_verdict("The passage supports it.") == "UNPARSED"
    assert parse_verdict("") == "UNPARSED"
    assert parse_verdict("Answer: SUPPORTED") == "UNPARSED"
