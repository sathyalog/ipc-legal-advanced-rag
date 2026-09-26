"""Grounded answer generation with verified citations.

The model must return quotes copied from the sections it was given. Each quote
is checked against the source text; quotes that don't match are dropped, and an
answer left with no verifiable citation is flagged. This moves "don't
hallucinate" from a prompt instruction to a check the code enforces.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from ipc_rag.config import ROOT, Settings, get_settings
from ipc_rag.generation.llm import Usage, get_answer_llm
from ipc_rag.retrieval.retriever import IPCRetriever, RetrievalResult

NOT_FOUND = "I cannot find this in the provided IPC text."


class Citation(BaseModel):
    section_id: str = Field(description="IPC section number, e.g. '302' or '498A'")
    quote: str = Field(description="Exact span copied from that section's text")


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = Field(default=False, description="True if the sections do not answer the question")


@dataclass
class AnswerResult:
    question: str
    answer: str
    abstained: bool
    citations: list[Citation]
    dropped_citations: list[Citation]
    retrieval: RetrievalResult
    usage: Usage | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def grounded(self) -> bool:
        return self.abstained or bool(self.citations)

    @property
    def contexts(self) -> list[str]:
        return [r.section.text for r in self.retrieval.sections]


@lru_cache
def load_prompt(version: str) -> str:
    return (ROOT / "prompts" / f"{version}.md").read_text()


def _norm(text: str) -> str:
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_citations(citations: list[Citation], retrieval: RetrievalResult) -> tuple[list[Citation], list[Citation]]:
    texts = {r.section.section_id: _norm(r.section.text) for r in retrieval.sections}
    ok, dropped = [], []
    for c in citations:
        sid = c.section_id.upper().replace("SECTION", "").replace(" ", "").strip(".")
        src, quote = texts.get(sid), _norm(c.quote)
        # exact substring, or near-exact (tolerates a changed quote mark / dropped bracket)
        if src and quote and (quote in src or fuzz.partial_ratio(quote, src) >= 95):
            ok.append(Citation(section_id=sid, quote=c.quote))
        else:
            dropped.append(c)
    return ok, dropped


def build_user_message(question: str, retrieval: RetrievalResult) -> str:
    parts = ["<sections>"]
    for r in retrieval.sections:
        s = r.section
        status = "repealed/omitted" if s.repealed else "in force"
        parts.append(f'<section id="{s.section_id}" title="{s.title}" chapter="{s.chapter_no} - {s.chapter_title}" '
                     f'pages="{s.page_start}-{s.page_end}" status="{status}">\n{s.text}\n</section>')
    parts.append("</sections>")
    if retrieval.refs.unknown:
        parts.append(f"Note: the user referred to section(s) {', '.join(retrieval.refs.unknown)}, "
                     "which do not exist in this edition of the Code.")
    parts.append(f"<question>{question}</question>")
    return "\n".join(parts)


class IPCAssistant:
    def __init__(self, settings: Settings | None = None, retriever: IPCRetriever | None = None):
        self.s = settings or get_settings()
        if retriever is None:
            rewriter = None
            if self.s.query_rewrite:
                from ipc_rag.generation.rewrite import QueryRewriter
                rewriter = QueryRewriter(self.s)
            retriever = IPCRetriever(self.s, rewriter=rewriter)
        self.retriever = retriever
        self.system = load_prompt(self.s.prompt_version)

    def ask(self, question: str) -> AnswerResult:
        t = time.perf_counter()
        retrieval = self.retriever.search(question)
        timings = {"retrieval": (time.perf_counter() - t) * 1000, **retrieval.timings_ms}
        if not retrieval.sections:
            return AnswerResult(question, NOT_FOUND, True, [], [], retrieval, timings_ms=timings)

        t = time.perf_counter()
        parsed, usage = get_answer_llm(self.s).generate(self.system, build_user_message(question, retrieval), Answer)
        timings["generation"] = (time.perf_counter() - t) * 1000

        ok, dropped = verify_citations(parsed.citations, retrieval)
        text = parsed.answer.strip() or NOT_FOUND
        return AnswerResult(question, text, parsed.abstained, ok, dropped, retrieval, usage, timings)
