"""LLM backends behind one tiny interface: `generate(system, user, schema) -> (parsed, usage)`.

- ollama    : local model, $0 - default for development
- anthropic : Claude via the official SDK - for final / benchmark runs
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol, TypeVar

from pydantic import BaseModel

from ipc_rag.config import Settings, get_settings

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMBackend(Protocol):
    model: str

    def generate(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]: ...


class AnthropicBackend:
    def __init__(self, model: str):
        import anthropic

        self.model = model
        self._client = anthropic.Anthropic()

    def generate(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=4096,
            # the system prompt is identical on every call -> cache it
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            raise RuntimeError(f"Claude returned no parsable answer (stop_reason={resp.stop_reason})")
        return resp.parsed_output, Usage(self.model, resp.usage.input_tokens, resp.usage.output_tokens)


class OllamaBackend:
    def __init__(self, model: str, base_url: str, context_window: int):
        from llama_index.llms.ollama import Ollama

        self.model = model
        self._llm = Ollama(model=model, base_url=base_url, temperature=0.0, request_timeout=300,
                           context_window=context_window, json_mode=True)

    def generate(self, system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
        from llama_index.core.llms import ChatMessage

        sllm = self._llm.as_structured_llm(schema)
        resp = sllm.chat([ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)])
        raw = resp.raw if isinstance(resp.raw, schema) else schema.model_validate_json(resp.message.content)
        usage = resp.additional_kwargs or {}
        return raw, Usage(self.model, int(usage.get("prompt_eval_count", 0) or 0), int(usage.get("eval_count", 0) or 0))


@lru_cache
def _backend(provider: str, model: str, base_url: str, context_window: int) -> LLMBackend:
    return AnthropicBackend(model) if provider == "anthropic" else OllamaBackend(model, base_url, context_window)


def get_answer_llm(settings: Settings | None = None) -> LLMBackend:
    s = settings or get_settings()
    model = s.anthropic_answer_model if s.llm_provider == "anthropic" else s.ollama_model
    return _backend(s.llm_provider, model, s.ollama_base_url, s.ollama_context_window)
