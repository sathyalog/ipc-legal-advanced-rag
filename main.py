"""Streamlit UI - a thin client over the ipc_rag package.

    streamlit run main.py

If no LLM is reachable the app still works in "sections only" mode: it shows
the retrieved IPC sections without a generated answer.
"""

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import streamlit as st

from ipc_rag.config import get_settings

st.set_page_config(page_title="IPC Legal AI Assistant", page_icon="⚖️", layout="centered")
settings = get_settings()


@st.cache_resource(show_spinner="Loading index and models…")
def load_assistant():
    from ipc_rag.generation.answer import IPCAssistant
    from ipc_rag.store import get_qdrant_client
    # The Qdrant index is gitignored, so fresh deploys (Streamlit Cloud, Spaces) must build it on first start
    if not settings.sections_path.exists() or not get_qdrant_client().collection_exists(settings.collection):
        from ipc_rag.ingest.__main__ import run as ingest
        if ingest() != 0:
            raise RuntimeError("Ingestion failed - see logs")
    return IPCAssistant(settings)


@st.cache_data(ttl=300, show_spinner=False)
def llm_available() -> bool:
    """Cheap check so deploys without an LLM default to sections-only mode instead of erroring."""
    if settings.llm_provider == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    import httpx
    try:
        return httpx.get(settings.ollama_base_url, timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def render_sections(result, cited: set[str]) -> None:
    for r in result.sections:
        s = r.section
        mark = "📌 " if s.section_id in cited else ""
        how = "exact match" if r.source == "exact" else f"search · {r.score:.2f}"
        with st.expander(f"{mark}Section {s.section_id} — {s.title}  ·  p.{s.page_start}  ·  {how}"):
            st.caption(f"Chapter {s.chapter_no}: {s.chapter_title}" + ("  ·  **repealed/omitted**" if s.repealed else ""))
            st.write(s.text)


st.title("⚖️ Indian Penal Code (IPC) AI Assistant")
st.caption("Answers grounded in the text of the IPC, 1860, with verified section citations.")

with st.sidebar:
    st.subheader("Settings")
    provider = settings.llm_provider
    model = settings.anthropic_answer_model if provider == "anthropic" else settings.ollama_model
    st.markdown(f"**LLM:** `{provider}` · `{model}`")
    has_llm = llm_available()
    answer_mode = st.toggle("Generate answer (uses LLM)", value=has_llm,
                            help="Off = show retrieved sections only (no LLM needed)")
    if not has_llm:
        st.caption("No LLM reachable - showing retrieved sections only.")
    st.markdown(f"**Retrieval:** router + hybrid (bge + BM25) + rerank · top {settings.rerank_top_n}")
    if st.button("Clear chat"):
        st.session_state.messages = []

assistant = load_assistant()
st.session_state.setdefault("messages", [])

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if question := st.chat_input("Ask about an IPC section or offence, e.g. 'What is the punishment for stalking?'"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not answer_mode:
            with st.spinner("Searching…"):
                result = assistant.retriever.search(question)
            text = "Retrieved sections: " + ", ".join(f"**{i}**" for i in result.section_ids) if result.sections \
                else "No matching sections found."
            st.markdown(text)
            render_sections(result, set())
        else:
            try:
                with st.spinner("Searching and drafting answer…"):
                    res = assistant.ask(question)
            except Exception as e:  # LLM down / no key: degrade to retrieval-only instead of a stack trace
                if type(e).__name__ == "AuthenticationError":
                    hint = "Anthropic rejected the API key - check ANTHROPIC_API_KEY in .env (console.anthropic.com)."
                elif provider == "ollama":
                    hint = f"Is Ollama running? `brew services start ollama` and `ollama pull {settings.ollama_model}`."
                else:
                    hint = f"{type(e).__name__}: {str(e)[:200]}"
                st.error(f"LLM unavailable - showing retrieved sections only. {hint}")
                result = assistant.retriever.search(question)
                render_sections(result, set())
                text = "(LLM unavailable)"
            else:
                text = res.answer
                if res.citations:
                    text += "\n\n**Sources:** " + "; ".join(f"Section {c.section_id} — “{c.quote}”" for c in res.citations)
                elif not res.abstained:
                    text += "\n\n⚠️ _No citation could be verified against the source text — treat with caution._"
                st.markdown(text)
                render_sections(res.retrieval, {c.section_id for c in res.citations})
                timing = " · ".join(f"{k} {v / 1000:.1f}s" for k, v in res.timings_ms.items())
                st.caption(timing + (f" · tokens {res.usage.input_tokens}/{res.usage.output_tokens}" if res.usage else ""))
    st.session_state.messages.append({"role": "assistant", "content": text})
