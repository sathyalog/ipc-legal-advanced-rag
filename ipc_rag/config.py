"""Central configuration. Every tunable lives here and can be overridden via env / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
# Export .env into os.environ too, so SDKs that read their own variables (ANTHROPIC_API_KEY) see them.
load_dotenv(ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="IPC_", extra="ignore")

    # --- source document ---
    pdf_path: Path = ROOT / "assets" / "IPC_186045.pdf"
    sections_path: Path = ROOT / "data" / "sections.jsonl"  # parsed-section docstore (source of truth)

    # --- vector store (embedded Qdrant, no server needed) ---
    qdrant_path: Path = ROOT / ".qdrant"
    qdrant_url: str | None = None  # set to use a Qdrant server / cloud instead of local mode
    collection: str = "ipc_sections_v1"

    # --- models (all local + free by default) ---
    embed_model: str = "BAAI/bge-base-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    # MiniLM beat bge-reranker-base on the golden set (lay recall 0.56 vs 0.50) at ~1/6 of the latency
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_device: str = "cpu"  # MPS stalls when mixed with onnxruntime (fastembed) in one process

    # --- chunking ---
    child_chunk_tokens: int = 400  # sections longer than this are split into child nodes

    # --- retrieval ---
    hybrid_top_k: int = 20  # candidates from dense+sparse fusion
    rerank_top_n: int = 5  # sections handed to the LLM
    use_reranker: bool = True
    use_router: bool = True  # deterministic section-number lookup
    query_rewrite: bool = False  # LLM restates lay questions in IPC terms (extra LLM call per query)

    # --- LLM ---
    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_model: str = "qwen2.5:7b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    # KV-cache memory grows with this; prompts here are ~2-5k tokens, so 8k is plenty (16k OOMs an 8 GB Mac)
    ollama_context_window: int = 8192
    anthropic_answer_model: str = "claude-sonnet-5"
    anthropic_judge_model: str = "claude-haiku-4-5"
    prompt_version: str = "answer_v1"


@lru_cache
def get_settings() -> Settings:
    return Settings()
