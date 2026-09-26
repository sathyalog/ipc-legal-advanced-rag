import pytest

from ipc_rag.retrieval.router import parse_section_refs

KNOWN = {"2", "10", "34", "120A", "120B", "299", "300", "302", "304", "304B", "376", "376D", "420", "498", "498A"}


@pytest.mark.parametrize("query, expected", [
    ("What is IPC section 302?", ["302"]),
    ("Explain Section 498A", ["498A"]),
    ("what is 498-A IPC", ["498A"]),
    ("punishment under s. 120B", ["120B"]),
    ("u/s 376 D", ["376D"]),
    ("Is 420 IPC bailable?", ["420"]),
    ("section 302 and 34 of IPC", ["302", "34"]),
    ("ss. 299, 300 & 304B", ["299", "300", "304B"]),
    ("difference between 299 and 300", ["299", "300"]),
    ("302 vs 304", ["302", "304"]),
    ("302", ["302"]),
    ("what is 498A?", ["498A"]),
    ("explain 120-B", ["120B"]),
    ("section 302 A person was killed", ["302"]),  # trailing "A" is not part of the id
])
def test_extracts_section_ids(query, expected):
    assert parse_section_refs(query, KNOWN).ids == expected


@pytest.mark.parametrize("query", [
    "punishment for murder",
    "imprisonment of 10 years for 2 persons",
    "imprisonment between 2 and 5 years",
    "fine of 100 and 200 rupees",
    "what does article 21 say",  # Constitution, not IPC
])
def test_ignores_numbers_that_are_not_sections(query):
    assert parse_section_refs(query, KNOWN).ids == []


def test_unknown_section_is_reported_not_guessed():
    refs = parse_section_refs("What does section 376DB say?", KNOWN)
    assert refs.ids == [] and refs.unknown == ["376DB"]
