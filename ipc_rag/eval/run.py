"""Experiment runner.

    python -m ipc_rag.eval.run retrieval                       # all configs, both suites
    python -m ipc_rag.eval.run retrieval --configs full --limit 100
    python -m ipc_rag.eval.run ragas --configs full            # needs an LLM (see ipc_rag.eval.ragas_eval)
    python -m ipc_rag.eval.run ragas --no-llm --configs dense,hybrid,full   # retrieval tier, $0

Results go to reports/ as CSV + markdown so the README can quote real numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path

from ipc_rag.config import ROOT, get_settings
from ipc_rag.ingest.nodes import load_sections
from ipc_rag.eval.retrieval_suite import evaluate, lookup_cases, title_cases

REPORTS = ROOT / "reports"

# name -> IPCRetriever kwargs.  "legacy" is the old LangChain setup, recreated.
RETRIEVAL_CONFIGS: dict[str, dict | None] = {
    "legacy": None,
    "dense": dict(mode="dense", use_router=False, use_reranker=False),
    "sparse": dict(mode="sparse", use_router=False, use_reranker=False),
    "hybrid": dict(mode="hybrid", use_router=False, use_reranker=False),
    "hybrid_rerank": dict(mode="hybrid", use_router=False, use_reranker=True),
    "full": dict(mode="hybrid", use_router=True, use_reranker=True),
    "full_rewrite": dict(mode="hybrid", use_router=True, use_reranker=True, rewrite=True),  # needs an LLM
}


def make_retriever(name: str, **overrides):
    from ipc_rag.retrieval.retriever import IPCRetriever

    kwargs = dict(RETRIEVAL_CONFIGS[name] or {})
    if kwargs.pop("rewrite", False):
        from ipc_rag.generation.rewrite import QueryRewriter
        kwargs["rewriter"] = QueryRewriter()
    return IPCRetriever(**kwargs, **overrides)


def _searcher(name: str, sections):
    if RETRIEVAL_CONFIGS[name] is None:
        from ipc_rag.eval.legacy_baseline import LegacyRetriever
        return LegacyRetriever(sections).search
    r = make_retriever(name, top_n=10)
    return lambda q: r.search(q).section_ids


def write_table(rows: list[dict], stem: str, title: str) -> Path:
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    md = [f"# {title}", "", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
          "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    md += ["| " + " | ".join(str(v) for v in r.values()) + " |" for r in rows]
    path = REPORTS / f"{stem}.md"
    path.write_text("\n".join(md) + "\n")
    return path


def run_retrieval(configs: list[str], suites: list[str], limit: int | None, seed: int = 7) -> None:
    sections = load_sections(get_settings().sections_path)
    ordered = list(sections.values())
    all_cases = {"lookup": lookup_cases(ordered, all_templates=False), "title": title_cases(ordered)}
    rows, failures = [], {}
    for name in configs:
        search = _searcher(name, sections)
        for suite in suites:
            cases = all_cases[suite]
            if limit and limit < len(cases):
                cases = random.Random(seed).sample(cases, limit)
            res = evaluate(name, suite, cases, search)
            rows.append(res.row())
            failures[f"{name}/{suite}"] = res.failures
            print(res.row(), flush=True)
    path = write_table(rows, "retrieval_results", "Retrieval benchmark (deterministic, no LLM)")
    (REPORTS / "retrieval_failures.json").write_text(json.dumps(failures, indent=1, ensure_ascii=False))
    print(f"\nwrote {path}")


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--configs", default=",".join(RETRIEVAL_CONFIGS))
    r.add_argument("--suites", default="lookup,title")
    r.add_argument("--limit", type=int, default=None, help="random sample per suite (for quick runs)")
    g = sub.add_parser("ragas")
    g.add_argument("--configs", default="full")
    g.add_argument("--dataset", default=str(ROOT / "data" / "eval" / "golden.jsonl"))
    g.add_argument("--limit", type=int, default=None)
    g.add_argument("--no-llm", action="store_true", help="retrieval-tier metrics only ($0, no LLM needed)")
    args = ap.parse_args()

    if args.cmd == "retrieval":
        run_retrieval(args.configs.split(","), args.suites.split(","), args.limit)
    else:
        from ipc_rag.eval.ragas_eval import run_ragas
        run_ragas(args.configs.split(","), Path(args.dataset), args.limit, with_llm=not args.no_llm)


if __name__ == "__main__":
    main()
