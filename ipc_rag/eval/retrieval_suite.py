"""Deterministic retrieval benchmark - no LLM, no API cost, runs in CI.

Two query sets are generated from the parsed Code itself, so they cover
*every* section instead of a hand-picked few:

- lookup: "What does IPC section 302 say?" style queries (3 phrasings per section).
          The router must answer these with 100% hit@1.
- title:  the section's own heading with the number removed
          ("Punishment for murder" -> 302). Tests the semantic path.

Metrics: hit@1, hit@5 (= recall@5 with one relevant section), MRR@10.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ipc_rag.ingest.pdf_parser import Section

LOOKUP_TEMPLATES = [
    "What does IPC section {id} say?",
    "Punishment under s. {id} IPC",
    "Explain Section {id} of the Indian Penal Code",
]


@dataclass(frozen=True)
class Case:
    query: str
    expected: str  # section_id


@dataclass
class SuiteResult:
    config: str
    suite: str
    n: int
    hit1: float
    hit5: float
    mrr: float
    p50_ms: float
    p95_ms: float
    failures: list[tuple[str, str, list[str]]]

    def row(self) -> dict:
        return {"config": self.config, "suite": self.suite, "n": self.n, "hit@1": round(self.hit1, 4),
                "hit@5": round(self.hit5, 4), "mrr@10": round(self.mrr, 4),
                "p50_ms": round(self.p50_ms), "p95_ms": round(self.p95_ms)}


def lookup_cases(sections: Iterable[Section], all_templates: bool = True) -> list[Case]:
    out = []
    for i, s in enumerate(sections):
        templates = LOOKUP_TEMPLATES if all_templates else [LOOKUP_TEMPLATES[i % len(LOOKUP_TEMPLATES)]]
        out += [Case(t.format(id=s.section_id), s.section_id) for t in templates]
    return out


def title_cases(sections: Iterable[Section]) -> list[Case]:
    out = []
    for s in sections:
        title = s.title.strip("“”\" .")
        # skip repealed stubs and one-word definitions ("Gender", "Number") - not meaningful search targets
        if s.repealed or len(title.split()) < 3:
            continue
        out.append(Case(title, s.section_id))
    return out


def evaluate(config: str, suite: str, cases: list[Case], search: Callable[[str], list[str]],
             progress: bool = True) -> SuiteResult:
    """`search(query) -> ranked section ids`."""
    hit1 = hit5 = 0
    rr: list[float] = []
    lat: list[float] = []
    failures = []
    for n, c in enumerate(cases, 1):
        t = time.perf_counter()
        ranked = search(c.query)[:10]
        lat.append((time.perf_counter() - t) * 1000)
        rank = ranked.index(c.expected) + 1 if c.expected in ranked else None
        hit1 += rank == 1
        hit5 += bool(rank and rank <= 5)
        rr.append(1 / rank if rank else 0.0)
        if rank != 1:
            failures.append((c.query, c.expected, ranked[:5]))
        if progress and n % 100 == 0:
            print(f"  [{config}/{suite}] {n}/{len(cases)} hit@1={hit1 / n:.3f}", flush=True)
    lat.sort()
    return SuiteResult(config, suite, len(cases), hit1 / len(cases), hit5 / len(cases), statistics.fmean(rr),
                       lat[len(lat) // 2], lat[int(len(lat) * 0.95) - 1], failures)
