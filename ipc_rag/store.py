"""Shared, lazily-created heavy resources: embedding model, Qdrant client, vector store."""

from __future__ import annotations

import atexit
from functools import lru_cache

from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ipc_rag.config import get_settings

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache
def get_embed_model(model_name: str | None = None) -> HuggingFaceEmbedding:
    name = model_name or get_settings().embed_model
    return HuggingFaceEmbedding(
        model_name=name,
        query_instruction=BGE_QUERY_INSTRUCTION if "bge" in name.lower() else None,
        normalize=True,
    )


@lru_cache
def get_qdrant_client() -> QdrantClient:
    s = get_settings()
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url)
    # Embedded mode: data on disk, no server. Only one process may open it at a time.
    s.qdrant_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(s.qdrant_path))
    atexit.register(client.close)  # release the local-mode file lock cleanly
    return client


def rrf_fusion(dense: VectorStoreQueryResult, sparse: VectorStoreQueryResult, alpha: float = 0.5,
               top_k: int = 20, k: int = 60) -> VectorStoreQueryResult:
    """Reciprocal Rank Fusion: score = sum(1 / (k + rank)).

    Rank-based, so it doesn't care that BM25 scores and cosine similarities
    live on different scales (the default relative-score fusion does care).
    """
    scores: dict[str, float] = {}
    nodes: dict[str, object] = {}
    for weight, result in ((alpha, dense), (1 - alpha, sparse)):
        for rank, node in enumerate(result.nodes or []):
            scores[node.node_id] = scores.get(node.node_id, 0.0) + 2 * weight / (k + rank + 1)
            nodes.setdefault(node.node_id, node)
    ranked = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return VectorStoreQueryResult(nodes=[nodes[i] for i in ranked], similarities=[scores[i] for i in ranked], ids=ranked)


def get_vector_store(collection: str | None = None) -> QdrantVectorStore:
    s = get_settings()
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection or s.collection,
        enable_hybrid=True,
        fastembed_sparse_model=s.sparse_model,
        hybrid_fusion_fn=rrf_fusion,
        batch_size=64,
    )
