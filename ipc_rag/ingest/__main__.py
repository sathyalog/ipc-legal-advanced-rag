"""Build the index:  python -m ipc_rag.ingest [--rebuild]

Steps: parse PDF -> completeness gate -> save section store -> nodes -> embed
(dense + BM25 sparse) -> upsert into Qdrant. Re-running is idempotent: node IDs
are deterministic and the ingestion docstore skips unchanged nodes.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.storage.docstore import SimpleDocumentStore

from ipc_rag.config import get_settings
from ipc_rag.ingest.nodes import save_sections, sections_to_nodes
from ipc_rag.ingest.pdf_parser import parse_sections
from ipc_rag.store import get_embed_model, get_qdrant_client, get_vector_store

log = logging.getLogger("ipc_rag.ingest")


def run(rebuild: bool = False) -> int:
    s = get_settings()
    t0 = time.perf_counter()

    sections, toc, report = parse_sections(s.pdf_path)
    log.info("TOC entries: %d | sections parsed: %d", report.toc_count, report.parsed_count)
    if not report.ok:
        # Completeness gate: refuse to build an index that is known to be missing sections.
        log.error("Missing sections (in TOC, not found in body): %s", report.missing)
        return 1
    save_sections(sections, s.sections_path)

    nodes = sections_to_nodes(sections, s.child_chunk_tokens)
    log.info("Retrieval units (nodes): %d from %d sections", len(nodes), len(sections))

    docstore_path = s.sections_path.parent / "ingest_docstore.json"
    client = get_qdrant_client()
    if rebuild:
        if client.collection_exists(s.collection):
            client.delete_collection(s.collection)
        docstore_path.unlink(missing_ok=True)

    docstore = SimpleDocumentStore.from_persist_path(str(docstore_path)) if docstore_path.exists() else SimpleDocumentStore()
    pipeline = IngestionPipeline(
        transformations=[get_embed_model()],
        vector_store=get_vector_store(),
        docstore=docstore,
        docstore_strategy=DocstoreStrategy.UPSERTS,  # hash-based: unchanged nodes are skipped
    )
    written = pipeline.run(nodes=nodes, show_progress=True)
    docstore.persist(str(docstore_path))

    count = client.count(s.collection).count
    log.info("Upserted %d changed nodes | collection '%s' now has %d points | %.1fs",
             len(written), s.collection, count, time.perf_counter() - t0)
    if count != len(nodes):
        log.error("Point count %d != node count %d (stale points?) - rerun with --rebuild", count, len(nodes))
        return 1
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true", help="drop the collection and re-embed everything")
    sys.exit(run(ap.parse_args().rebuild))


if __name__ == "__main__":
    main()
