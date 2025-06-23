# 🧠 Autonomous AI Insight Engine

A mini autonomous LLM agent that processes user research queries, retrieves context from a vector store, and returns structured, source-cited, hallucination-free answers — with conversational memory, caching, real-time streaming, and SQL logging.

## 🚀 Features

- ✅ LLM-powered structured answers  
- ✅ Embedding-based retrieval (Chroma + Ollama)  
- ✅ FastAPI endpoint (`POST /query`)  
- ✅ LangGraph with memory: remembers prior messages  
- ✅ Cached answers for identical/similar queries (no re-retrieval)  
- ✅ Real-time token streaming (`/query/stream`)  
- ✅ PostgreSQL logging of every query + answer  
- ✅ Source citation and hallucination control  

## 📁 Project Structure

```
.
├── main.py               # FastAPI app
├── llm_agent.py          # LangGraph agent logic, memory, generation
├── retriever.py          # Embedding + Chroma vector DB
├── db.py                 # SQLAlchemy + PostgreSQL logging setup
├── data/                 # Folder with .txt/.pdf research docs
├── requirements.txt      # Project dependencies
├── README.md             # You're here
└── example_response.json # Sample query + result
```

## 🧪 Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

(Ensure Ollama is installed and PostgreSQL is running)

### 2. PostgreSQL Setup

Make sure a PostgreSQL server is running locally, then create the DB:

```sql
CREATE DATABASE perceivenow_db;
```

Update your `DATABASE_URL` in `db.py` like:

```python
DATABASE_URL = "postgresql+psycopg2://postgres:your_password@localhost:5432/perceivenow_db"
```

### 3. Add documents

Put at least 10 `.txt` or `.pdf` files in the `data/` folder.

### 4. Run the app

```bash
uvicorn main:app --reload
```

Open Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

## 📮 API Usage

### `POST /query`

**Request:**
```json
{
  "query": "What are the top 3 use cases for GraphQL in enterprise SaaS?"
}
```

**Response:**
```json
{
  "answer": "- Efficient data fetching\n- Developer productivity...\n[Source: doc_01.txt]",
  "sources": ["doc_01.txt", "doc_03.pdf"]
}
```

### `POST /query/stream`

Same input as `/query` but returns tokens one-by-one using Server-Sent Events (SSE). Useful for real-time UIs.

## 🧠 Memory, Caching & Logging

This engine uses LangGraph’s `MemorySaver` with per-thread memory (`thread_id`) so it remembers past interactions. If a user asks the same or a related question, the answer will be served from memory — without re-triggering embedding or retrieval.

Every query + generated answer + sources are logged into a **PostgreSQL table `query_logs`**, with timestamps for tracking and analytics.

## 🧱 Powered By

- [LangGraph](https://github.com/langchain-ai/langgraph)  
- [LangChain](https://github.com/langchain-ai/langchain)  
- [Ollama](https://ollama.com)  
- [ChromaDB](https://www.trychroma.com)  
- [PostgreSQL](https://www.postgresql.org/)  

## ✅ Bonus

- Token-by-token streaming  
- Memory-optimized retrieval  
- PostgreSQL history logging  
- Ready for frontend integration

## 👨‍💻 Author

Built with love for the Perceive Now technical challenge 💡
