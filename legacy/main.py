import os

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # hide unnecessary logs

from chunking import chunk_data
from database import create_index
from document_loading import load_document
from dotenv import load_dotenv
from embeddings import insert_or_fetch_embeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from pinecone import Pinecone
import streamlit as st

# 1. Load Environment Variables
load_dotenv(override=True)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
connect_database = create_index()
print(pc.list_indexes())

# 2. Configure Streamlit Page Layout
st.set_page_config(
    page_title="IPC Legal AI Assistant",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("⚖️ Indian Penal Code (IPC) AI Assistant")
if "documents" not in st.session_state:
    st.session_state.documents = None
st.caption(
    "Query Indian Penal Code sections and offenses with document-grounded answers."
)

# 3. Initialize Model and Runnable Chain
model = ChatOpenRouter(
    model="openrouter/free", openrouter_api_key=os.getenv("OPENROUTER_API_KEY")
)

system_prompt_text = """You are a precise, document-grounded AI assistant. Your sole purpose is to answer the user's question using ONLY the provided document context.

Instructions:
- Always mention the relevant IPC Section numbers (e.g., IPC Section 212) explicitly.
- Fetch the exact text from the document that answers the user's question.
### CORE GROUNDING RULES:
1. Zero External Knowledge: You must operate ONLY on the information explicitly contained within the provided context. Do NOT use your prior knowledge, general training data, or outside assumptions.
2. Zero Inferences: Do NOT assume, extrapolate, speculate, or deduce facts that are not directly stated in the context.
3. Strict Fallback: If the exact information needed to answer the query is missing, incomplete, or ambiguous in the context, you MUST output EXACTLY: "I cannot find this information in the provided document."

### CONTEXT:
{context}
User Question:
{input}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_text),
        ("human", "{input}"),
    ]
)

chain = prompt | model

# 4. Initialize Chat History in Streamlit Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize DuckDuckGo tool
search_tool = DuckDuckGoSearchRun()

with st.sidebar:
    st.header("🔍 Quick DuckDuckGo web search(not IPC)")
    with st.form("sidebar_search_form"):
        duckduckgo_search = st.text_input(
            "Search widget",
            key="placeholder",
        )
        search_submitted = st.form_submit_button("Search Web")

    if search_submitted and duckduckgo_search.strip():
        with st.spinner("Searching..."):
            try:
                results = search_tool.invoke(duckduckgo_search)
                st.text(results)
            except Exception as e:
                st.error(f"Search failed: {e}")

    st.header("Upload the IPC_186045 document")
    file = "assets/IPC_186045.pdf"
    if st.button("Upload & Index Document"):
        with st.spinner("Loading & indexing document into pinecone..."):
            try:
                st.session_state.documents = load_document(file)
                st.success("Document uploaded successfully!")

                if st.session_state.documents:
                    total_pages = len(st.session_state.documents)
                    st.text(f"Total pages: {total_pages}")
                    chunks = chunk_data(st.session_state.documents)

                    # Clean chunk metadata
                    for chunk in chunks:
                        chunk.metadata = {
                            k: v
                            for k, v in chunk.metadata.items()
                            if isinstance(v, (str, int, float, bool))
                            and k not in ["dl_meta", "doc_items"]
                        }

                    st.session_state.chunks = chunks
                    st.text(f"Total chunks: {len(st.session_state.chunks)}")

                    vector_store = insert_or_fetch_embeddings(chunks)
                    st.session_state.vector_store = vector_store
                    st.success(f"Indexed {len(chunks)} chunks into Pinecone!")
                else:
                    st.error("No documents loaded.")
            except Exception as e:
                st.error(f"Failed to process document: {e}")


def generate_response(user_input):
    if st.session_state.get("vector_store") is None:
        st.session_state.vector_store = insert_or_fetch_embeddings()

    context_text = ""
    docs = []

    if st.session_state.vector_store:
        # Increase k to retrieve more context, removing rigid metadata filters
        retriever = st.session_state.vector_store.as_retriever(
            search_kwargs={
                "k": 8
            }
        )
        
        # Expand user queries with substantive legal keywords to bypass index matches
        expanded_query = user_input
        if "328" in user_input:
            expanded_query = f"Section 328 Causing hurt by means of poison drug stupefying unwholesome {user_input}"

        docs = retriever.invoke(expanded_query)

        print(f"Step 3 - Retrieved Docs: {len(docs)}")
        print("\n--- RETRIEVED CHUNKS DEBUG ---")
        for i, doc in enumerate(docs):
            print(f"Chunk {i+1} Metadata: {doc.metadata}")
            print(f"Chunk {i+1} Sample Text: {doc.page_content[:150]}...\n")

        # Combine unique chunks while filtering out obvious Index/Arrangement of Sections lines
        valid_contents = []
        for doc in docs:
            # Skip chunks that are clearly just table-of-content headers
            if "ARRANGEMENT OF SECTIONS" in doc.page_content.upper() and len(doc.page_content) < 500:
                continue
            valid_contents.append(doc.page_content)

        context_text = "\n\n".join(valid_contents)

    # Stream response from model
    stream = chain.stream({"context": context_text, "input": user_input})
    for chunk in stream:
        yield chunk.content

    # Clean display for citations (without broken page numbers)
    if docs:
        page_numbers = set()
        for doc in docs:
            page = doc.metadata.get("page") or doc.metadata.get("page_number")
            # Only include valid positive page numbers
            if page is not None and isinstance(page, (int, float)) and page > 0:
                page_numbers.add(str(int(page)))

        if page_numbers:
            pages_str = ", ".join(sorted(page_numbers, key=lambda x: int(x)))
            ref_msg = f"\n\n---\n📌 **Reference Source:** IPC PDF Document (Page(s): {pages_str})"
        else:
            ref_msg = "\n\n---\n📌 **Reference Source:** IPC Document Context"

        yield ref_msg


# 6. Chat Input Loop
if user_prompt := st.chat_input("Ask a question about the Indian Penal Code..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(generate_response(user_prompt))

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )