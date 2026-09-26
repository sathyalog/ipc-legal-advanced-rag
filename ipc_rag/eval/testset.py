"""Synthetic test-set generation with RAGAS (knowledge-graph based).

    IPC_LLM_PROVIDER=anthropic python -m ipc_rag.eval.testset --size 60

RAGAS builds a knowledge graph over the sections (headlines, keyphrases,
entities, similarity links) and synthesizes single-hop and multi-hop questions
with reference answers. Output goes to data/eval/synthetic_candidates.jsonl -
REVIEW IT before merging into golden.jsonl: synthetic questions are a starting
point, and an unreviewed golden set just measures the generator's mistakes.

Uses RAGAS's LlamaIndex integration (TestsetGenerator.from_llama_index), so
the same LlamaIndex LLM / embedding objects drive generation.
"""

from __future__ import annotations

import argparse
import json
import re

from ipc_rag.config import ROOT, get_settings
from ipc_rag.ingest.nodes import load_sections

OUT = ROOT / "data" / "eval" / "synthetic_candidates.jsonl"


def _generator_llm():
    s = get_settings()
    if s.llm_provider == "anthropic":
        from llama_index.llms.anthropic import Anthropic
        return Anthropic(model=s.anthropic_judge_model, max_tokens=4096)
    from llama_index.llms.ollama import Ollama
    return Ollama(model=s.ollama_model, base_url=s.ollama_base_url, temperature=0.0, request_timeout=300,
                  context_window=s.ollama_context_window)


def main() -> None:
    from llama_index.core import Document
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers import default_query_distribution

    from ipc_rag.store import get_embed_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=60)
    ap.add_argument("--min-chars", type=int, default=400, help="skip tiny definitional sections")
    args = ap.parse_args()

    s = get_settings()
    sections = [x for x in load_sections(s.sections_path).values() if not x.repealed and len(x.text) >= args.min_chars]
    docs = [Document(text=x.text, metadata={"section_id": x.section_id, "title": x.title}) for x in sections]

    gen = TestsetGenerator.from_llama_index(llm=_generator_llm(), embedding_model=get_embed_model(),
                                            llm_context="Indian Penal Code, 1860 - questions a lawyer or citizen would ask")
    dataset = gen.generate_with_llamaindex_docs(docs, testset_size=args.size,
                                                query_distribution=default_query_distribution(gen.llm))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for i, row in enumerate(dataset.to_pandas().to_dict("records"), 1):
            contexts = row.get("reference_contexts") or []
            ids = sorted({m.group(1) for c in contexts if (m := re.match(r"(\d{1,3}[A-Z]{0,3})\.", c.strip()))})
            f.write(json.dumps({"id": f"s{i:03d}", "type": row.get("synthesizer_name", "synthetic"),
                                "question": row["user_input"], "reference": row.get("reference", ""),
                                "reference_section_ids": ids, "needs_review": True}, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} - review before adding to golden.jsonl")


if __name__ == "__main__":
    main()
