import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from database import create_index

load_dotenv(override=True)
PC_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PC_API_KEY)
index_name = "ipc186045"

def insert_or_fetch_embeddings(documents=None):
    pinecone_index = pc.Index(index_name)
    # 1. Initialize embeddings configured for OpenRouter
    # embeddings = OpenAIEmbeddings(
    #     model="openai/text-embedding-3-small",  # OpenRouter model ID
    #     openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    #     dimensions=1536,
    #     openai_api_base="https://openrouter.ai/api/v1",  # Crucial for OpenRouter
    #     check_embedding_ctx_length=False,  # Prevents tiktoken context errors
    # )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if documents:
        return PineconeVectorStore.from_documents(
            documents=documents, embedding=embeddings, index_name=index_name
        )

    if index_name in pc.list_indexes().names():
        print(f"Index {index_name} already exists. Loading embeddings..")
        vector_store = PineconeVectorStore(index=pinecone_index, embedding=embeddings)
        print(f"OK")
        return vector_store
    else:
        print(f"Index {index_name} does not exist. Creating index..")
        create_index()
        vector_store = PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
        return vector_store

