"""Turn parsed sections into LlamaIndex nodes (small-to-big layout).

Every section becomes one or more *retrieval units* in the vector store:
- short sections -> a single node holding the whole section
- long sections (s.300, s.375 ...) -> several child nodes split at sentence
  boundaries, each tagged with the parent `section_id`

At query time hits are collapsed back to their `section_id` and the LLM
receives the *full* section text from the section store, so an answer never
sees half a section.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode, TextNode

from ipc_rag.ingest.pdf_parser import Section

_NS = uuid.UUID("6f1c6a2e-8f0e-4b8e-9d3c-1f0a2b3c4d5e")
_EMBED_EXCLUDED = ["page_start", "page_end", "repealed", "has_illustrations", "has_explanation", "chunk_index", "chapter_no"]


def node_id(section_id: str, chunk: int) -> str:
    """Deterministic UUID, so re-ingesting upserts instead of duplicating (a bug in the legacy app)."""
    return str(uuid.uuid5(_NS, f"ipc:{section_id}:{chunk}"))


def sections_to_nodes(sections: list[Section], child_chunk_tokens: int = 400) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=child_chunk_tokens, chunk_overlap=40)
    nodes: list[TextNode] = []
    for s in sections:
        meta = {
            "section_id": s.section_id,
            "title": s.title,
            "chapter_no": s.chapter_no,
            "chapter_title": s.chapter_title,
            "page_start": s.page_start,
            "page_end": s.page_end,
            "repealed": s.repealed,
            "has_illustrations": s.has_illustrations,
            "has_explanation": s.has_explanation,
        }
        chunks = splitter.split_text(s.text) or [s.text]
        for i, chunk in enumerate(chunks):
            node = TextNode(
                id_=node_id(s.section_id, i),
                text=chunk,
                metadata={**meta, "chunk_index": i},
                excluded_embed_metadata_keys=_EMBED_EXCLUDED,
                excluded_llm_metadata_keys=_EMBED_EXCLUDED,
                # "Section 302 | Punishment for murder | OF OFFENCES AFFECTING THE HUMAN BODY" is
                # prepended to what gets embedded, so every child chunk knows its section.
                metadata_template="{key}: {value}",
                text_template="{metadata_str}\n\n{content}",
            )
            nodes.append(node)
    return nodes


def embed_text(node: TextNode) -> str:
    return node.get_content(metadata_mode=MetadataMode.EMBED)


# ------------------------------------------------------------ section store
def save_sections(sections: list[Section], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def load_sections(path: Path) -> dict[str, Section]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python -m ipc_rag.ingest` first")
    out: dict[str, Section] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            s = Section(**json.loads(line))
            out[s.section_id] = s
    return out
