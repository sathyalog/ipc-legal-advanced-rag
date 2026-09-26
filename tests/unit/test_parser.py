"""Parser tests run against the real PDF - the parser *is* the data contract."""

import pytest

from ipc_rag.config import get_settings
from ipc_rag.ingest.pdf_parser import parse_sections, section_sort_key


@pytest.fixture(scope="module")
def parsed():
    pdf = get_settings().pdf_path
    if pdf.stat().st_size < 10_000:
        pytest.skip("PDF is a Git LFS pointer - run `git lfs pull`")
    return parse_sections(pdf)


def test_completeness_gate_every_toc_section_found(parsed):
    sections, toc, report = parsed
    assert report.ok, f"missing: {report.missing}"
    assert report.parsed_count == report.toc_count == len(toc)


def test_sections_in_code_order_and_unique(parsed):
    ids = [s.section_id for s in parsed[0]]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids, key=section_sort_key)


@pytest.mark.parametrize("sid, title_start, text_contains", [
    ("302", "Punishment for murder", "shall be punished with death"),
    ("328", "Causing hurt by means of poison", "stupefying, intoxicating or unwholesome drug"),
    ("498A", "Husband or relative of husband", "cruelty"),
    ("120B", "Punishment of criminal conspiracy", "party to a criminal conspiracy"),
    ("77", "Act of Judge when acting judicially", "Judge when acting judicially"),  # non-bold heading in PDF
    ("29A", "“Electronic record”", "electronic record"),
])
def test_known_sections(parsed, sid, title_start, text_contains):
    sec = {s.section_id: s for s in parsed[0]}[sid]
    assert sec.title.startswith(title_start)
    assert text_contains in sec.text
    assert sec.text.startswith(f"{sid}.")


def test_footnotes_and_superscripts_are_stripped(parsed):
    sec = {s.section_id: s for s in parsed[0]}
    # page-15 footnote "1. The Indian Penal Code has been extended to Berar..." must not leak into s.1
    assert "Berar" not in sec["1"].text
    # amendment superscript "9[4. Extension" -> heading starts cleanly
    assert sec["4"].text.startswith("4. Extension of Code")


def test_repealed_sections_flagged(parsed):
    sec = {s.section_id: s for s in parsed[0]}
    assert sec["13"].repealed and not sec["302"].repealed


def test_chapter_metadata(parsed):
    sec = {s.section_id: s for s in parsed[0]}
    assert (sec["302"].chapter_no, sec["302"].chapter_title) == ("XVI", "OF OFFENCES AFFECTING THE HUMAN BODY")
    assert sec["498A"].chapter_no == "XXA"
