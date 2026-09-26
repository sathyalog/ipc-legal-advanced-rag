"""Re-creates the *old* app's retrieval so the before/after numbers are apples to apples.

Legacy pipeline (see legacy/): whole-page text -> RecursiveCharacterTextSplitter(1000, 200)
-> all-MiniLM-L6-v2 -> dense top-8, TOC pages included. The only difference is
that it runs in a local Qdrant collection instead of Pinecone, and uses PyMuPDF
text instead of Docling (both produce the page text).

Because legacy chunks don't know which section they belong to, a chunk counts
as a hit for section X if it contains X's heading or an 80-char span of X's text.
"""

from __future__ import annotations

import re
import uuid
from functools import lru_cache

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ipc_rag.config import get_settings
from ipc_rag.ingest.pdf_parser import Section
from ipc_rag.store import get_embed_model, get_qdrant_client

COLLECTION = "ipc_legacy_baseline"
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_norm = re.compile(r"[^a-z]+")


def _n(text: str) -> str:
    # letters only: raw page text is full of amendment superscripts ("shall 3[extend") that the
    # parsed section text doesn't have, so digits/brackets would break substring matching
    return _norm.sub("", text.lower())


def build(force: bool = False) -> None:
    client = get_qdrant_client()
    if client.collection_exists(COLLECTION) and not force:
        return
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    doc = pymupdf.open(get_settings().pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""])
    nodes = []
    for pno, page in enumerate(doc, 1):
        for i, chunk in enumerate(splitter.split_text(page.get_text())):
            nodes.append(TextNode(id_=str(uuid.uuid5(uuid.NAMESPACE_URL, f"legacy:{pno}:{i}")), text=chunk,
                                  metadata={"page": pno}))
    store = QdrantVectorStore(client=client, collection_name=COLLECTION)
    VectorStoreIndex(nodes, storage_context=StorageContext.from_defaults(vector_store=store),
                     embed_model=get_embed_model(MODEL), show_progress=True)


class LegacyRetriever:
    def __init__(self, sections: dict[str, Section], k: int = 8):
        build()
        store = QdrantVectorStore(client=get_qdrant_client(), collection_name=COLLECTION)
        self._r = VectorStoreIndex.from_vector_store(store, embed_model=get_embed_model(MODEL)).as_retriever(similarity_top_k=k)
        self._sections = {sid: _n(s.text) for sid, s in sections.items()}
        self._heads = {sid: _n(s.text[:90]) for sid, s in sections.items()}  # title + start of body (TOC lines lack the body)

    @lru_cache(maxsize=4096)
    def _chunk_sections(self, chunk: str) -> tuple[str, ...]:
        c = _n(chunk)
        mid = c[len(c) // 2 - 30: len(c) // 2 + 30]
        return tuple(sid for sid in self._sections
                     if (len(self._heads[sid]) > 30 and self._heads[sid] in c) or (len(mid) >= 60 and mid in self._sections[sid]))

    def search(self, query: str) -> list[str]:
        ranked: list[str] = []
        for hit in self._r.retrieve(query):
            for sid in self._chunk_sections(hit.node.get_content()):
                if sid not in ranked:
                    ranked.append(sid)
        return ranked
