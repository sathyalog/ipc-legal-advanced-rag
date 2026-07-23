from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st
def chunk_data(data, chunk_size=1000):
    chunk_size = st.session_state.get("chunk_size") or 1000
    chunk_overlap = st.session_state.get("chunk_overlap") or 200
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap),
    # Force splits at legal double newlines or major structural markers:
    separators=["\n\n", "\n", ". ", " ", ""])
    chunks = text_splitter.split_documents(data)
    print(f"DEBUG: Generated {len(chunks)} text chunks.")
    print(f"Step 2 - Total Chunks: {len(chunks)}")
    if len(chunks) == 0:
        raise ValueError("Text splitter returned 0 chunks! Check your chunk size and overlap settings.")
    return chunks