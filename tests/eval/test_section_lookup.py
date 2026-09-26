"""The "100%" guarantee, enforced: every section, every phrasing, hit@1.

Uses the real section store and the full retriever config. Pure lookups are
answered by the router without touching the vector index, so this is fast.
"""

import pytest

from ipc_rag.config import get_settings
from ipc_rag.eval.retrieval_suite import evaluate, lookup_cases


@pytest.fixture(scope="module")
def retriever():
    if not get_settings().sections_path.exists():
        pytest.skip("run `python -m ipc_rag.ingest` first")
    from ipc_rag.retrieval.retriever import IPCRetriever
    return IPCRetriever(mode="hybrid", use_router=True, use_reranker=True)


def test_every_section_lookup_is_hit_at_1(retriever):
    cases = lookup_cases(retriever.sections.values(), all_templates=True)
    res = evaluate("full", "lookup", cases, lambda q: retriever.search(q).section_ids, progress=False)
    assert len(cases) == 3 * len(retriever.sections)
    assert res.hit1 == 1.0, f"{len(res.failures)} misses, e.g. {res.failures[:5]}"
