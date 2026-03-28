# 🤖 Multi-Agent RAG Chatbot — Master Plan

> **Tech Stack:** Python · Streamlit · Groq (LLaMA-3.1) · LangChain · LangGraph · ChromaDB · SQLite

---

## 📁 Project Structure

```
chatboart 22/
├── app.py                    # Streamlit UI entry point
├── config.py                 # Centralised settings from .env
├── requirements.txt          # All Python dependencies
├── setup.bat                 # One-click virtual environment setup (Windows)
├── .env                      # Your API keys (DO NOT share)
├── .env.example              # Template for .env
│
├── rag/
│   ├── document_loader.py    # PDF → chunks (PyPDFLoader + splitter)
│   ├── vector_store.py       # ChromaDB + HuggingFace embeddings
│   └── retriever.py          # Hybrid search: Semantic + BM25
│
├── agents/
│   ├── base_agent.py         # Abstract interface all agents implement
│   ├── router_agent.py       # 🚦 LLM intent classification + rule fallback
│   ├── math_agent.py         # 🔢 Python eval + LLM chain-of-thought
│   ├── rag_agent.py          # 🔍 Hybrid retrieval + grounded LLM answer
│   └── memory_agent.py       # 🧠 SQLite recall + personalised response
│
├── orchestrator/
│   └── graph.py              # LangGraph StateGraph (all agents wired together)
│
├── memory/
│   └── sqlite_memory.py      # Persistent conversation history (SQLite)
│
├── utils/
│   └── logger.py             # Structured logging → logs/chatbot.log
│
└── data/
    ├── chroma_db/            # ChromaDB vector store (auto-created)
    └── chat_memory.db        # SQLite conversation history (auto-created)
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│              LangGraph Orchestrator          │
│                                             │
│   ┌──────────────┐                          │
│   │ Router Agent │  LLM intent + rule       │
│   │  🚦          │  fallback classification  │
│   └──────┬───────┘                          │
│          │ route: math | rag | memory        │
│   ┌──────┴────────────────────┐             │
│   ▼           ▼               ▼             │
│ Math Agent  RAG Agent   Memory Agent        │
│ 🔢          🔍           🧠                  │
│   └──────┬────────────────────┘             │
│          ▼                                  │
│   Save to SQLite Memory                     │
└─────────────────────────────────────────────┘
    │
    ▼
Streamlit UI (token-level streaming)
```

### RAG Pipeline Flow
```
PDF Upload
  → PyPDFLoader (extract pages)
  → RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
  → HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  → ChromaDB (persist to ./data/chroma_db)

Query
  → Semantic Search (ChromaDB cosine similarity)
  + BM25 Keyword Search (rank-bm25)
  → Reciprocal Rank Fusion (RRF)
  → Top-K chunks → Groq LLM → Grounded answer
```

---

## 🚀 Setup (Step by Step)

### Step 1 — Run the setup script
```bat
cd "d:\personal\my prosnal workkk\chatboart 22"
setup.bat
```
This will:
- Create `.venv` virtual environment
- Install all packages from `requirements.txt`
- Create `data/` directories

### Step 2 — Activate environment
```bat
.venv\Scripts\activate
```

### Step 3 — Verify your .env
Your `.env` should contain:
```env
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL_NAME=llama-3.1-8b-instant
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_DB_PATH=./data/chroma_db
SQLITE_DB_PATH=./data/chat_memory.db
```

### Step 4 — Run the chatbot
```bat
streamlit run app.py
```
Open your browser → `http://localhost:8501`

---

## 🧪 Testing the Agents

| What to ask | Expected Agent | Example |
|---|---|---|
| Arithmetic | 🔢 Math Agent | `"What is 144 / 12?"` |
| Complex math | 🔢 Math Agent | `"If a train goes 80km/h for 2.5h, how far?"` |
| Document content | 🔍 RAG Agent | `"Summarise the uploaded PDF"` |
| Specific facts | 🔍 RAG Agent | `"What does the document say about X?"` |
| Recall | 🧠 Memory Agent | `"What did we discuss earlier?"` |
| History | 🧠 Memory Agent | `"What was my last question?"` |

---

## 🔑 Key Technical Decisions

| Component | Choice | Why |
|---|---|---|
| LLM | LLaMA-3.1-8b-instant (Groq) | Fast, free, strong reasoning |
| Embeddings | all-MiniLM-L6-v2 | Lightweight, high semantic quality |
| Retrieval | Semantic + BM25 (RRF fusion) | Best accuracy — catches meaning AND keywords |
| Math safety | AST-based eval, not built-in eval() | Secure, no code injection risk |
| Router | LLM + keyword fallback | Never fails, always routes |
| Memory | SQLite | Lightweight, zero-server, persistent |
| Streaming | Word-by-word via Groq | Low latency feel in the UI |

---

## 📊 Evaluation Targets

| Metric | Target | How to Measure |
|---|---|---|
| Retrieval Accuracy | ≥ 92% top-3 relevance | Upload test PDF, ask factual questions |
| Response Latency | ≤ 30% faster than baseline | Compare with/without streaming |
| Memory Recall | ≥ 45% personalisation improvement | Ask same question after 5+ turns |

---

## 🔧 Extending the System

### Add a new agent
1. Create `agents/new_agent.py` extending `BaseAgent`
2. Implement `run(query, context, history, session_id) -> str`
3. Add node in `orchestrator/graph.py`
4. Add route label in `config.py`
5. Update `RouterAgent` prompt + rules

### Change the LLM model
In `.env`:
```env
GROQ_MODEL_NAME=llama-3.1-70b-versatile  # for higher quality
```

### Adjust retrieval chunk size
In `.env`:
```env
CHUNK_SIZE=750      # larger = more context per chunk
CHUNK_OVERLAP=100   # larger = smoother chunk boundaries
TOP_K_RETRIEVAL=8   # more chunks = higher recall
```

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY not set` | Add key to `.env` |
| `No documents uploaded yet` | Upload a PDF via sidebar first |
| `ChromaDB connection error` | Check `data/chroma_db/` folder exists |
| `Import error` | Make sure `.venv` is activated |
| `streamlit: command not found` | Run `pip install streamlit` inside `.venv` |

---

## 📝 Logs

Logs are written to `logs/chatbot.log` — check here for debugging:
```bat
type logs\chatbot.log
```

---

*Built with ❤️ using Python, Streamlit, Groq, LangChain, LangGraph, ChromaDB & SQLite*
