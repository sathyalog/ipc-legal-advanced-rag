# 📚 Indian Penal Code (IPC) Question-Answering System

An Advanced Retrieval-Augmented Generation (RAG) legal assistant built with **LangChain**, **Streamlit**, and **Pinecone** to query Indian Penal Code (IPC) sections with strict document-grounded accuracy.

---

## 📌 1. Project Overview

Large Language Models (LLMs) often suffer from knowledge cutoffs and hallucinations when asked about specialized legal documents. This project implements a **RAG pipeline** that stores Indian Penal Code text chunks inside a Pinecone vector database. When a user queries the system, it retrieves relevant context chunks and forces the LLM to answer using *only* that factual context.

### 📸 Project Verification & Screenshots

#### 1. RAG Agent in Action
The AI assistant searches vector embeddings in Pinecone to return grounded legal answers along with exact document citations:

![final RAG](<Screenshot 2026-07-23 at 5.35.29 PM.png>)

#### 2. Source Document Verification
Cross-verify the AI response and page citations against the original PDF source text:

![PDF](<Screenshot 2026-07-23 at 5.35.53 PM.png>)

#### 3. DuckDuckGo Web Search Integration
A secondary search tool is embedded in the sidebar for general web queries outside the IPC PDF context:
![final RAG](<Screenshot 2026-07-23 at 5.35.29 PM.png>)


---

## 🚀 2. Steps to Setup and Run

Follow these steps to set up and run the project locally using `uv` (or standard `pip`).

### Step 1: Prerequisites & Package Manager
If you don't have `uv` installed, install it using one of the following methods:

* **macOS / Linux:**
  ```bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

###  Step 2: Environment Sync
Sync project dependencies and set up your virtual environment:
`uv sync`

###  Step 3: Configure Environment Variables
Create a .env file in the project root directory and add your secret API keys:
PINECONE_API_KEY=your_pinecone_api_key
OPENROUTER_API_KEY=your_openrouter_api_key

### Step 4: Run the Application
Launch the Streamlit interface:
`uv run streamlit run main.py`

Once project runs for first time, open the sidebar and click on "Upload Document" button.

For additional developer guidelines, check out the [Developer Notes](NOTES.md)

### 🛠️ 3. Project Execution & RAG Pipeline
How the Pipeline Works Step-by-Step
	1.	Document Ingestion: The IPC PDF document (assets/IPC_186045.pdf) is loaded into memory using DoclingLoader. 
    2.	Chunking & Metadata Sanitization: The text is broken down into small, digestible section chunks. Any non-serializable metadata (like dl_meta) is stripped out to ensure clean upserts.
    3.	Vector Embedding & Storage: Chunks are embedded into high-dimensional vectors and stored inside a Pinecone vector index (database.py).
    4.	Smart Query Expansion: User queries (e.g., "IPC 328") are expanded with substantive legal keywords (e.g., "Section 328 Causing hurt by means of poison...") to bypass table-of-contents noise and target full legal text chunks.
	5.	Grounded Answer Generation: LangChain retrieves the top matching chunks and passes them to the LLM. The system prompt forces strict adherence to context and appends verified PDF page citations.

### 🧠 4. Tech Architecture & Key Concepts
🛠️ Key Technologies Used
•	LangChain: Connects the LLM with external vector storage and manages prompt execution flows using LCEL (LangChain Expression Language).
•	Pinecone: Fully-managed cloud vector database used to store high-dimensional embeddings and perform high-speed similarity searches.
•	OpenRouter / OpenAI: LLM providers generating grounded responses based purely on retrieved context.
•	Streamlit: Powers the frontend user interface and chat environment.

### 🗄️ SQL vs. Vector Databases (Pinecone)
Feature	Traditional SQL Database	Vector Database (Pinecone)
Search Mechanism	Exact Keyword / Key Match	Semantic Similarity (Mathematical Vector Distance)
Storage Engine	SQL tables / Local .sqlite files	Cloud-managed Serverless Vector Index
Example Query	SELECT * FROM ipc WHERE section = 378;	index.query(vector=query_emb, top_k=3, filter={"statute": "IPC"})
Output	Exact row match	Ranked list of nearest contextual matches with metadata

Embedding Vectors & Mathematical Distance
Text embeddings convert unstructured natural language into numerical representations. Vector databases store these embeddings and compute geometric distance (e.g., Cosine Similarity) to find relevant document chunks.

### 💡 5. Extra Notes, Tips & Developer Reference
🤖 ReAct Agents & LangChain Tools
A ReAct Agent combines Reasoning (thought process) and Acting (tool execution) to choose actions dynamically:
•	Code Execution: Uses Python REPL for complex math or algorithm generation.
•	General Knowledge / Historical Info: Uses Wikipedia or DuckDuckGo for web results.

### ⚡ Caching & Streamlining
To optimize application response times and reduce API overhead, prompt responses and model calls can be cached using LangChain caching mechanisms:
from langchain_core.caches import InMemoryCache, SQLiteCache
from langchain_core.globals import set_llm_cache

#### In-Memory Cache
set_llm_cache(InMemoryCache())

#### Persistent SQLite Cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

### 🔗 LangChain Expression Language (LCEL)
Instead of legacy chain constructors, this project uses modern LCEL syntax (chain = prompt | model) with ChatPromptTemplate to stream results dynamically using chain.stream().

