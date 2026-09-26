---
title: IPC Legal AI Assistant
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.64.0
app_file: main.py
pinned: false
---

# IPC Legal AI Assistant — LlamaIndex + Qdrant + RAGAS

Question answering over the **Indian Penal Code, 1860** (112-page India Code PDF, 566 sections), with answers grounded in the statute text and citations verified against the source.

The first version (LangChain + Pinecone, kept in [`legacy/`](legacy/)) answered roughly 80% of questions correctly. This rebuild treats retrieval as an engineering problem with a test suite. Every section lookup is checked in CI, and everything else is measured with RAGAS.

## Results

<!-- RESULTS:START -->
**Section lookups: 1,698 / 1,698 correct at rank 1** (566 sections × 3 phrasings). This is enforced by `tests/eval/test_section_lookup.py`, which runs in about 20 s with no LLM.

Deterministic retrieval benchmark (no LLM, generated from the Code itself). The *lookup* suite has 566 queries of the form "What does IPC section N say?". The *title* suite has 466 queries: the section heading with its number removed.

| Configuration | Lookup hit@1 | Lookup hit@5 | Title hit@1 | Title hit@5 |
|---|---|---|---|---|
| **Legacy** (1000-char chunks, MiniLM, dense top-8) | 0.2% | 1.2% | 35.2% | 81.1% |
| Dense only (bge-base, section-aware nodes) | 20.3% | 37.3% | 97.4% | 100% |
| Sparse only (BM25) | 60.6% | 68.9% | 93.8% | 99.8% |
| Hybrid (dense + BM25, RRF) | 41.5% | 73.3% | 97.6% | 100% |
| **Full** (router + hybrid + rerank) | **100%** ¹ | **100%** ¹ | 100% ² | 100% ² |

¹ All 1,698 lookup queries (3 phrasings per section). ² Random sample of 60 title queries; the full-size run is pending.

What the table shows:
- **No embedding model retrieves sections by number.** Dense search found only 20% of lookups, even on clean section-aware nodes. Numbers need an exact path, which is why the router exists.
- **Chunking was the bigger problem for meaning-based queries.** Moving from 1000-char chunks to one node per section took title hit@1 from 35% to 97%.
- The legacy ~81% hit@5 on titles matches the "right about 80% of the time" experience with the old app.

RAGAS retrieval tier on the 40-question golden set (`IDBasedContextRecall`, no LLM):

| Configuration | All | Lookup | Semantic | Lay-language | Multi-section |
|---|---|---|---|---|---|
| Dense | 0.765 | 0.833 | 1.00 | 0.375 | 0.75 |
| Hybrid | 0.824 | 1.00 | 1.00 | 0.375 | 0.75 |
| Hybrid + `bge-reranker-base` | 0.868 | 1.00 | 1.00 | 0.50 | 0.875 |
| Hybrid + `ms-marco-MiniLM-L-6-v2` (default) | 0.868 | 1.00 | 1.00 | **0.56** | 0.75 |

MiniLM was chosen over bge-reranker-base because it scored higher on lay-language questions at about 1/6 of the CPU latency (~1 s vs ~7 s per query). **Lay-language questions ("my neighbour attacked me with a knife") are the known weak spot.** The optional LLM query rewrite (`IPC_QUERY_REWRITE=true`) targets them; it is not measured yet.

The LLM-judged RAGAS tier (faithfulness, answer relevancy, factual correctness, abstention) is built but has not run yet. Run `uv run ipc-eval ragas --configs full,full_rewrite` with an LLM configured.
<!-- RESULTS:END -->

## Why the old version missed ~20%

| Root cause (legacy) | Fix (this version) |
|---|---|
| Dense-only search with `all-MiniLM-L6-v2`; embeddings can't match identifiers like "302" or "498A" | **Deterministic router**: section references are parsed with regex and fetched by ID. That path can't miss. |
| 1000-char splitter cut section headings away from their text | **Structure-aware parser**: one node per section, using PDF layout (bold headings, font sizes) |
| TOC pages (every section number and title) outranked the actual text | TOC is used only as a **completeness checklist**, never indexed |
| Page footnotes ("1. Subs. by Act 4 of 1898…") look like sections | Footnotes dropped using the footnote rule's y-position; 7pt amendment superscripts stripped |
| No section metadata, so no exact lookup or filtering | Every node carries `section_id`, `title`, `chapter`, `pages`, `repealed` |
| Re-indexing duplicated vectors | Deterministic UUIDs + LlamaIndex `IngestionPipeline` docstore (upserts; a re-run writes 0 nodes) |
| No reranking, no evaluation | Hybrid (bge dense + BM25 sparse, RRF) → cross-encoder rerank, and a two-tier eval harness |
| Hardcoded query hack for section 328 | Removed; the router generalises to all 566 sections |

## What runs where

```
Your question
   ├─ 1. Router (regex)           "section 328" → fetch section 328 directly      ← local
   ├─ 2. Hybrid search (Qdrant)   meaning (bge) + keywords (BM25), fused by RRF    ← local, .qdrant/ on disk
   ├─ 3. Rerank (cross-encoder)   re-scores the top 20, keeps the best 5           ← local, CPU
   └─ 4. LLM answer               structured answer + quotes, then verified        ← Claude API or Ollama
```

**Retrieval (steps 1–3) needs no API key and no internet.** The index and models live on your machine. The LLM is used only to write the answer from the retrieved sections, so it can't cite a section that retrieval didn't find. If the LLM is unavailable (no key, bad key, Ollama not running), the app still works and shows the retrieved sections with their full text. In the UI, "exact match" means the router found the section number. "search · -7.7" is the reranker's relevance score: higher is better, and negative values are normal.

## Architecture

```
PDF ──► pdf_parser (layout-aware) ──► completeness gate (TOC == parsed?) ──► sections.jsonl (source of truth)
                                                                                 │
                                          nodes (small-to-big: long sections → child chunks with section_id)
                                                                                 │
                                                     IngestionPipeline ─► Qdrant (dense bge-base + sparse BM25)

query ──► router ──(section ids)──────────────────────────────► exact sections (score 1.0)
            └──► [optional LLM rewrite] ─► hybrid search (RRF) ─► cross-encoder rerank ─► collapse chunks → full sections
                                                                                 │
                              Claude / Ollama structured output {answer, citations[{section_id, quote}], abstained}
                                                                                 │
                                        citation verifier (quote must be in the section text, else dropped)
```

| Layer | Choice | Why |
|---|---|---|
| Parsing | PyMuPDF with layout signals | Bold, font size and the footnote rule give exact section boundaries |
| Framework | LlamaIndex (`IngestionPipeline`, `QdrantVectorStore`, `BaseRetriever`, `SentenceTransformerRerank`) | |
| Vector DB | Qdrant, **embedded mode** (`.qdrant/` on disk; no server needed) | Native dense + sparse hybrid, payload filters; `IPC_QDRANT_URL` switches to a server |
| Embeddings | `BAAI/bge-base-en-v1.5` (local) + `Qdrant/bm25` sparse (fastembed) | BM25 carries exact terms; bge carries meaning |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local, CPU) | Beat `bge-reranker-base` on lay questions at ~1/6 the latency |
| LLM | `ollama` (default, $0) or `anthropic` (Claude Sonnet 5 answers, Haiku 4.5 judge) | One env var: `IPC_LLM_PROVIDER` |
| Eval | RAGAS 0.4 + deterministic suites | See below |

## Evaluation

**Tier 0: deterministic, no LLM, $0** (`ipc_rag/eval/retrieval_suite.py`). Queries are generated from the Code itself, so every section is covered:
- *lookup*: "What does IPC section {id} say?" (3 phrasings × 566 sections). `tests/eval/test_section_lookup.py` fails the build unless **hit@1 = 100%**.
- *title*: the section heading without its number ("Punishment for murder" → 302). Tests the semantic path.
- Every configuration is compared, including a faithful re-creation of the legacy pipeline (`eval/legacy_baseline.py`).

**Tier 1: RAGAS retrieval metrics, no LLM, $0.** `IDBasedContextPrecision` / `IDBasedContextRecall` on the golden set (`data/eval/golden.jsonl`): 40 hand-written questions covering lookup, semantic, lay-language, multi-section and **unanswerable** cases, broken down per type.

**Tier 2: RAGAS answer metrics, needs an LLM.** `Faithfulness`, `AnswerRelevancy`, `ContextPrecision` (with reference), `ContextRecall`, `FactualCorrectness` (F1), a custom `DiscreteMetric` ("cites the section?"), plus deterministic **abstention accuracy** and **citation validity**. Judge calls are disk-cached (`.cache/ragas`), so re-runs only pay for changes.

**Growing the golden set:** `python -m ipc_rag.eval.testset --size 60` uses RAGAS `TestsetGenerator` (knowledge graph + single/multi-hop synthesizers) to draft candidates into `data/eval/synthetic_candidates.jsonl`. Review them before merging.

## Run it

Requires [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 and every pinned dependency from `uv.lock`.

```bash
git lfs pull                     # the PDF is stored in Git LFS
uv sync                          # create .venv with exact locked versions (add --no-dev to skip eval tools)
uv run ipc-ingest                # parse → completeness gate → embed → Qdrant  (~1 min, idempotent)
uv run pytest                    # unit tests + the 100% section-lookup gate
uv run ipc-app                   # Streamlit UI at http://localhost:8501
```

Evaluation:

```bash
uv run ipc-eval retrieval                                                  # tier 0 → reports/
uv run ipc-eval ragas --no-llm --configs dense,hybrid,hybrid_rerank,full   # tier 1, $0
uv run ipc-eval ragas --configs full,full_rewrite                          # tier 2, needs an LLM (below)
uv run ipc-testset --size 60                                               # draft new golden questions
```

An LLM is only needed for generated answers and the tier-2 metrics. Without one, the app still shows the retrieved sections. Choose based on your machine:

**Claude API (recommended; required on 8 GB machines).** Billed per token, about $0.01 per question with Sonnet 5. A Claude subscription does not include API credits.

```bash
cp .env.example .env        # .env is git-ignored. Never put keys in .env.example (it is committed)
# in .env:  IPC_LLM_PROVIDER=anthropic
#           ANTHROPIC_API_KEY=sk-ant-...      (console.anthropic.com)
```

**Ollama (free, local; needs 16 GB+ RAM for a 7B model).**

```bash
brew install ollama
brew services start ollama              # or `ollama serve` in its own terminal
ollama pull qwen2.5:7b-instruct         # ~4.7 GB; on 8 GB RAM use qwen2.5:3b-instruct and set IPC_OLLAMA_MODEL
```

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Whole machine freezes after asking a question | A local 7B model plus the app plus a browser exceeds 8 GB RAM, so macOS swaps | Use the Claude API, or a 3B Ollama model. `brew services stop ollama` frees the RAM |
| Terminal floods with `No module named 'torchvision'` | Streamlit's file watcher imports every lazy `transformers` submodule on each rerun | Already disabled in `.streamlit/config.toml` (`fileWatcherType = "none"`). Restart the app after code changes |
| "Anthropic rejected the API key" | Key missing from `.env`, revoked, or from another workspace | Put a valid key in `.env` (not `.env.example`) |
| "Storage folder … is already accessed by another instance" | Embedded Qdrant allows one process at a time | Stop the app before running evals, or set `IPC_QDRANT_URL` to a Qdrant server |

Dependencies live in `pyproject.toml` and are locked in `uv.lock`. Add one with `uv add <pkg>`. `requirements.txt` is generated from the lock only for Hugging Face Spaces, which reads that file.

Embedded Qdrant allows **one process at a time**. Stop Streamlit before running evals, or point `IPC_QDRANT_URL` at a Qdrant server.

## Layout

```
ipc_rag/
  config.py              all tunables (env-overridable, IPC_ prefix)
  store.py               embedding model, Qdrant client, RRF fusion
  ingest/                pdf_parser.py · nodes.py · __main__.py (pipeline + completeness gate)
  retrieval/             router.py (section refs) · retriever.py (router → hybrid → rerank → small-to-big)
  generation/            llm.py (Claude / Ollama) · answer.py (structured output + citation verifier) · rewrite.py
  eval/                  retrieval_suite.py · legacy_baseline.py · ragas_eval.py · testset.py · run.py
prompts/answer_v1.md     versioned system prompt
data/sections.jsonl      parsed Code (566 sections)      data/eval/golden.jsonl   golden set
tests/                   unit/ (parser, router, citations) · eval/ (100% lookup gate)
legacy/                  original LangChain + Pinecone app, kept for comparison
```
