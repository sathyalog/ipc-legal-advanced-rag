"""Generation-layer tests with a stub LLM - no model, no network."""

from ipc_rag.generation.answer import Answer, Citation, build_user_message, verify_citations
from ipc_rag.ingest.pdf_parser import Section
from ipc_rag.retrieval.retriever import RetrievalResult, RetrievedSection
from ipc_rag.retrieval.router import SectionRefs

S302 = Section("302", "Punishment for murder", "XVI", "OF OFFENCES AFFECTING THE HUMAN BODY", 72, 72,
               "302. Punishment for murder.—Whoever commits murder shall be punished with death or "
               "[imprisonment for life], and shall also be liable to fine.")


def _retrieval(unknown=()):
    return RetrievalResult("q", [RetrievedSection(S302, 1.0, "exact")], SectionRefs(["302"], list(unknown)))


def test_exact_quote_is_kept():
    ok, dropped = verify_citations([Citation(section_id="302", quote="shall be punished with death")], _retrieval())
    assert [c.section_id for c in ok] == ["302"] and not dropped


def test_quote_with_different_quote_marks_and_spacing_is_kept():
    ok, _ = verify_citations([Citation(section_id="Section 302", quote="Whoever  commits murder shall be")], _retrieval())
    assert ok and ok[0].section_id == "302"


def test_paraphrased_quote_is_dropped():
    ok, dropped = verify_citations([Citation(section_id="302", quote="murderers get the death penalty or life in jail")],
                                   _retrieval())
    assert not ok and len(dropped) == 1


def test_citation_to_section_not_retrieved_is_dropped():
    ok, dropped = verify_citations([Citation(section_id="300", quote="shall be punished with death")], _retrieval())
    assert not ok and dropped


def test_prompt_contains_sections_and_unknown_note():
    msg = build_user_message("What is 376DB?", _retrieval(unknown=["376DB"]))
    assert '<section id="302"' in msg and "376DB" in msg and "<question>" in msg


def test_answer_schema_defaults():
    a = Answer(answer="x")
    assert a.citations == [] and a.abstained is False
