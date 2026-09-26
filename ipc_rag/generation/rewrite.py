"""LLM query rewriting for lay-language questions.

"My neighbour attacked me with a knife" shares almost no words with
s.324 ("voluntarily causing hurt by dangerous weapons or means"). The
golden set shows this is where retrieval is weakest (lay questions), so an
optional step asks the LLM to restate the question in the Code's own
vocabulary. The rewrites are searched *in addition to* the original query and
fused, so a bad rewrite can add noise but cannot remove the original hits.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field

from ipc_rag.config import Settings, get_settings
from ipc_rag.generation.llm import get_answer_llm

SYSTEM = (
    "You translate questions about Indian criminal law into search queries for the text of the Indian Penal Code, "
    "1860. Use the Code's own terminology (e.g. 'voluntarily causing hurt', 'criminal intimidation', 'cheating', "
    "'right of private defence', 'grievous hurt', 'dangerous weapons'). Do not answer the question."
)


class Rewrite(BaseModel):
    queries: list[str] = Field(description="1-3 short search queries using IPC terminology", max_length=3)


class QueryRewriter:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()

    @lru_cache(maxsize=1024)
    def __call__(self, question: str) -> tuple[str, ...]:
        parsed, _ = get_answer_llm(self.s).generate(SYSTEM, f"Question: {question}", Rewrite)
        return tuple(q.strip() for q in parsed.queries[:3] if q.strip() and q.strip().lower() != question.lower())
