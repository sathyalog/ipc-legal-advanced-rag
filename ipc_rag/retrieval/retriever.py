"""IPC retriever: deterministic section lookup -> hybrid search -> rerank -> small-to-big.

    query ──► router ──(section ids found)──► exact fetch from section store (score 1.0)
                │
                └──► Qdrant hybrid (bge dense + BM25 sparse, RRF) top-20 retrieval units
                        └──► cross-encoder rerank ──► collapse chunks to sections ──► top-5 full sections

Every stage can be switched off, which is how the eval experiments compare
configurations (dense-only vs hybrid vs +rerank vs +router).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from functools import lru_cache
from collections.abc import Callable, Sequence
from typing import Literal

from llama_index.core import QueryBundle, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from ipc_rag.config import Settings, get_settings
from ipc_rag.ingest.nodes import load_sections
from ipc_rag.ingest.pdf_parser import Section
from ipc_rag.retrieval.router import SectionRefs, parse_section_refs
from ipc_rag.store import get_embed_model, get_vector_store

SearchMode = Literal["dense", "sparse", "hybrid"]
_MODES = {"dense": VectorStoreQueryMode.DEFAULT, "sparse": VectorStoreQueryMode.SPARSE, "hybrid": VectorStoreQueryMode.HYBRID}
_STOP = {"what", "is", "the", "of", "under", "section", "sections", "ipc", "a", "an", "and", "in", "explain",
         "punishment", "for", "s", "sec", "does", "say", "tell", "me", "about", "between", "difference", "to",
         "indian", "penal", "code", "provide", "provides", "deal", "with", "mean", "means"}


@dataclass
class RetrievedSection:
    section: Section
    score: float
    source: Literal["exact", "search"]


@dataclass
class RetrievalResult:
    query: str
    sections: list[RetrievedSection]
    refs: SectionRefs
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def section_ids(self) -> list[str]:
        return [r.section.section_id for r in self.sections]


@lru_cache
def _reranker(model: str, top_n: int, device: str) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(model=model, top_n=top_n, device=device)


def section_node(sec: Section) -> TextNode:
    return TextNode(
        id_=f"ipc-{sec.section_id}",
        text=sec.text,
        metadata={"section_id": sec.section_id, "title": sec.title, "chapter": f"{sec.chapter_no} - {sec.chapter_title}",
                  "pages": f"{sec.page_start}-{sec.page_end}"},
    )


def _fuse(result_lists: list[list[NodeWithScore]], k: int = 60) -> list[NodeWithScore]:
    """RRF across the original + rewritten queries (a no-op for a single list)."""
    if len(result_lists) == 1:
        return result_lists[0]
    scores: dict[str, float] = {}
    nodes: dict[str, NodeWithScore] = {}
    for hits in result_lists:
        for rank, h in enumerate(hits):
            scores[h.node.node_id] = scores.get(h.node.node_id, 0.0) + 1 / (k + rank + 1)
            nodes.setdefault(h.node.node_id, h)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)
    return [NodeWithScore(node=nodes[i].node, score=scores[i]) for i in ranked]


class IPCRetriever(BaseRetriever):
    def __init__(self, settings: Settings | None = None, *, mode: SearchMode = "hybrid",
                 use_router: bool | None = None, use_reranker: bool | None = None,
                 top_n: int | None = None, collection: str | None = None, embed_model: str | None = None,
                 rewriter: Callable[[str], Sequence[str]] | None = None):
        super().__init__()
        self.rewriter = rewriter  # optional: extra IPC-vocabulary queries for lay questions
        self.s = settings or get_settings()
        self.mode = mode
        self.use_router = self.s.use_router if use_router is None else use_router
        self.use_reranker = self.s.use_reranker if use_reranker is None else use_reranker
        self.top_n = top_n or self.s.rerank_top_n
        self.sections = load_sections(self.s.sections_path)
        self.known_ids = set(self.sections)
        index = VectorStoreIndex.from_vector_store(get_vector_store(collection), embed_model=get_embed_model(embed_model))
        k = self.s.hybrid_top_k
        self._vector = index.as_retriever(similarity_top_k=k, sparse_top_k=k, hybrid_top_k=k,
                                          vector_store_query_mode=_MODES[mode])

    # -- LlamaIndex interface: section-level nodes, usable in any query engine
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        res = self.search(query_bundle.query_str)
        return [NodeWithScore(node=section_node(r.section), score=r.score) for r in res.sections]

    def search(self, query: str) -> RetrievalResult:
        timings: dict[str, float] = {}
        refs = parse_section_refs(query, self.known_ids) if self.use_router else SectionRefs()
        picked: list[RetrievedSection] = [RetrievedSection(self.sections[i], 1.0, "exact") for i in refs.ids]

        # Pure lookups ("what is section 302?") need nothing else - extra sections are just noise.
        # Mixed queries ("is 302 applicable in self-defence?") also get semantic results.
        leftover = [w for w in re.findall(r"[a-z]+", query.lower()) if w not in _STOP and len(w) > 2]
        want_search = not picked or len(leftover) >= 3
        if want_search:
            queries = [query]
            if self.rewriter:
                t = time.perf_counter()
                queries += list(self.rewriter(query))
                timings["rewrite"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            hits = _fuse([self._vector.retrieve(q) for q in queries])
            timings["vector_search"] = (time.perf_counter() - t) * 1000
            if self.use_reranker and hits:
                t = time.perf_counter()
                hits = _reranker(self.s.rerank_model, len(hits), self.s.rerank_device).postprocess_nodes(hits, query_str=query)
                timings["rerank"] = (time.perf_counter() - t) * 1000
            budget = self.top_n if not picked else max(self.top_n - len(picked), 3)
            seen = {p.section.section_id for p in picked}
            extra: list[RetrievedSection] = []
            for h in hits:  # collapse child chunks -> parent section, keep best rank
                sid = h.node.metadata["section_id"]
                if sid not in seen:
                    seen.add(sid)
                    extra.append(RetrievedSection(self.sections[sid], float(h.score or 0.0), "search"))
                if len(extra) >= budget:
                    break
            picked += extra
        return RetrievalResult(query, picked, refs, timings)
