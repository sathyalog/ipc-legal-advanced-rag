"""RAGAS evaluation over the golden set (data/eval/golden.jsonl).

Two tiers, so the free part always runs:

1. Retrieval tier - no LLM, $0
   - IDBasedContextPrecision / IDBasedContextRecall (RAGAS), using section ids
     as context ids: exact, not judged by an LLM.

2. Answer tier - needs an answer LLM + judge LLM (IPC_LLM_PROVIDER=ollama|anthropic)
   RAGAS metrics (modern `ragas.metrics.collections` API):
   - Faithfulness            : are the answer's claims supported by the retrieved sections?
   - AnswerRelevancy         : does the answer address the question? (uses embeddings)
   - ContextPrecision (ref)  : are the relevant sections ranked at the top?
   - ContextRecall           : do the retrieved sections contain everything the reference needs?
   - FactualCorrectness (F1) : answer claims vs reference claims
   - cites_section (custom DiscreteMetric): does the answer name the IPC section(s)?
   Deterministic checks on the answer:
   - abstention accuracy     : abstains on unanswerable questions, answers the rest
   - citation validity       : share of model citations that survived verbatim verification

Judge calls are cached on disk (.cache/ragas), so re-runs only pay for what changed.
"""

from __future__ import annotations

import asyncio
import json
import math
import statistics
import time
import warnings
from collections import defaultdict
from pathlib import Path

from ipc_rag.config import ROOT, get_settings
from ipc_rag.eval.run import REPORTS, RETRIEVAL_CONFIGS, make_retriever, write_table

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

CITES_SECTION_PROMPT = (
    "You are grading an answer about the Indian Penal Code.\n"
    "Question: {user_input}\nAnswer: {response}\n"
    "Return 'pass' if the answer explicitly names the relevant IPC section number(s) (e.g. 'Section 302'), "
    "or if it correctly states that the question is not covered. Otherwise return 'fail'."
)


def load_golden(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[:limit] if limit else rows


# ------------------------------------------------------------------ judge setup
def build_judge():
    """RAGAS judge LLM + embeddings for the configured provider."""
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    s = get_settings()
    cache = DiskCacheBackend(cache_dir=str(ROOT / ".cache" / "ragas"))
    if s.llm_provider == "anthropic":
        from anthropic import AsyncAnthropic
        llm = llm_factory(s.anthropic_judge_model, provider="anthropic", client=AsyncAnthropic(), cache=cache,
                          max_tokens=4096)
    else:
        from openai import AsyncOpenAI  # Ollama exposes an OpenAI-compatible endpoint
        llm = llm_factory(s.ollama_model, provider="openai", cache=cache, max_tokens=4096,
                          client=AsyncOpenAI(base_url=f"{s.ollama_base_url}/v1", api_key="ollama"))
    emb = HuggingFaceEmbeddings(model=s.embed_model)
    return llm, emb


def build_metrics(llm, emb) -> dict:
    from ragas.metrics import DiscreteMetric
    from ragas.metrics.collections import (AnswerRelevancy, ContextPrecisionWithReference, ContextRecall,
                                           FactualCorrectness, Faithfulness)

    return {
        "faithfulness": (Faithfulness(llm=llm), ("user_input", "response", "retrieved_contexts")),
        "answer_relevancy": (AnswerRelevancy(llm=llm, embeddings=emb, strictness=1), ("user_input", "response")),
        "context_precision": (ContextPrecisionWithReference(llm=llm), ("user_input", "reference", "retrieved_contexts")),
        "context_recall": (ContextRecall(llm=llm), ("user_input", "retrieved_contexts", "reference")),
        "factual_correctness": (FactualCorrectness(llm=llm, mode="f1"), ("response", "reference")),
        "cites_section": (DiscreteMetric(name="cites_section", prompt=CITES_SECTION_PROMPT,
                                         allowed_values=["pass", "fail"]), ("user_input", "response", "llm")),
    }


async def _score(metric, fields: tuple[str, ...], sample: dict, llm) -> float | None:
    kwargs = {f: (llm if f == "llm" else sample[f]) for f in fields}
    try:
        res = await metric.ascore(**kwargs)
    except Exception as e:  # one bad judge call shouldn't sink the run
        print(f"    ! {type(metric).__name__}: {type(e).__name__}: {str(e)[:120]}")
        return None
    v = res.value
    if isinstance(v, str):
        return 1.0 if v == "pass" else 0.0
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


# ------------------------------------------------------------------ runner
def _id_metrics(retrieved: list[str], reference: list[str]) -> dict[str, float]:
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics import IDBasedContextPrecision, IDBasedContextRecall

    sample = SingleTurnSample(retrieved_context_ids=retrieved or ["<none>"], reference_context_ids=reference)
    return {"id_context_precision": asyncio.run(IDBasedContextPrecision().single_turn_ascore(sample)),
            "id_context_recall": asyncio.run(IDBasedContextRecall().single_turn_ascore(sample))}


def run_ragas(configs: list[str], dataset: Path, limit: int | None, with_llm: bool = True) -> None:
    golden = load_golden(dataset, limit)
    llm = emb = metrics = None
    if with_llm:
        from ipc_rag.generation.answer import IPCAssistant
        llm, emb = build_judge()
        metrics = build_metrics(llm, emb)

    per_row, summary = [], []
    for name in configs:
        if RETRIEVAL_CONFIGS.get(name) is None:
            raise SystemExit(f"config '{name}' is not an IPCRetriever config (legacy is retrieval-suite only)")
        retriever = make_retriever(name)
        assistant = IPCAssistant(retriever=retriever) if with_llm else None
        agg: dict[str, list[float]] = defaultdict(list)
        by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        tokens = [0, 0]
        t0 = time.perf_counter()

        for i, g in enumerate(golden, 1):
            row = {"config": name, "id": g["id"], "type": g["type"], "question": g["question"]}
            answerable = bool(g["reference_section_ids"])
            if with_llm:
                res = assistant.ask(g["question"])
                retrieved = res.retrieval.section_ids
                row.update(answer=res.answer, abstained=res.abstained,
                           citations=[c.section_id for c in res.citations],
                           dropped_citations=len(res.dropped_citations))
                if res.usage:
                    tokens[0] += res.usage.input_tokens
                    tokens[1] += res.usage.output_tokens
                row["abstention_correct"] = float(res.abstained != answerable)
                n_cit = len(res.citations) + len(res.dropped_citations)
                if n_cit:
                    row["citation_validity"] = len(res.citations) / n_cit
                if answerable:  # generation metrics are meaningless on questions that should be refused
                    sample = {"user_input": g["question"], "response": res.answer, "reference": g["reference"],
                              "retrieved_contexts": res.contexts}
                    for mname, (metric, fields) in metrics.items():
                        row[mname] = asyncio.run(_score(metric, fields, sample, llm))
            else:
                retrieved = retriever.search(g["question"]).section_ids

            row["retrieved"] = retrieved
            if answerable:
                row.update(_id_metrics(retrieved, g["reference_section_ids"]))
            for k, v in row.items():
                if isinstance(v, float) and k not in ("config",):
                    agg[k].append(v)
                    by_type[g["type"]][k].append(v)
            per_row.append(row)
            print(f"  [{name}] {i}/{len(golden)} {g['id']} ({g['type']})", flush=True)

        s = {"config": name, "n": len(golden)}
        s.update({k: round(statistics.fmean(v), 3) for k, v in sorted(agg.items())})
        if with_llm:
            s["answer_tokens_in/out"] = f"{tokens[0]}/{tokens[1]}"
        s["wall_s"] = round(time.perf_counter() - t0)
        summary.append(s)
        for typ, vals in by_type.items():
            summary.append({"config": f"{name} · {typ}", "n": sum(1 for g in golden if g["type"] == typ),
                            **{k: round(statistics.fmean(v), 3) for k, v in sorted(vals.items())}})

    REPORTS.mkdir(exist_ok=True)
    stem = "ragas_results" if with_llm else "ragas_retrieval_results"
    (REPORTS / f"{stem}_rows.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in per_row) + "\n")
    keys = list(dict.fromkeys(k for s in summary for k in s))
    path = write_table([{k: s.get(k, "") for k in keys} for s in summary], stem,
                       "RAGAS evaluation" + ("" if with_llm else " (retrieval tier only, no LLM)"))
    print(f"\nwrote {path}")
