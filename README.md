# Multi Agent Chatbot

Multi Agent Chatbot is a sophisticated AI-powered system designed for deep document reasoning and real-time information retrieval. Built on a modular, multi-agent architecture, it coordinates specialized LLM-driven agents to provide accurate, grounded, and up-to-date answers.

![Chatbot UI](C:\Users\kulde\.gemini\antigravity\brain\8c10ed46-8ce5-41d3-8b4e-edad6e824be9\final_verification_chatbot_1774686010530.png)

## 🛠 Features

*   **Multi-Agent Coordination:** Uses **LangGraph** to route queries across specialized nodes:
    *   **RAG Agent:** Retrieves and synthesizes knowledge from uploaded PDF documents.
    *   **Web Search Agent:** Uses real-time web search APIs for up-to-the-minute data.
    *   **Math Agent:** Handles complex calculations for higher accuracy.
    *   **Memory Agent:** Manages long-term conversational context across sessions.
*   **Context-Aware RAG:** A hybrid retrieval pipeline using **ChromaDB** and PDF parsing to ground all answers in source documents.
*   **Persistent SQLite Memory:** Unlike basic chatbots, this chatbot remembers past interactions using a custom SQLite-backed memory layer.
*   **Real-Time Streaming UI:** Fast, token-by-token response rendering with a premium dark-themed interface built using Vanilla JS and CSS.
*   **Automated Query Reformulation:** Reformulates natural language into optimized search strings before querying external APIs to ensure maximum precision.

## 🏗 Tech Stack

*   **Backend:** Python, Flask
*   **LLM Engine:** Groq (LLaMA-3.1-70B)
*   **Orchestration:** LangChain, LangGraph
*   **Database:** SQLite (Memory), ChromaDB (Vector Store)
*   **Frontend:** Vanilla JavaScript, HTML5, CSS3

## 📁 Project Structure

```text
├── agents/             # Modular agent logic (Math, Search, RAG, etc.)
├── api/                # Flask route handlers
├── memory/             # SQLite session persistence
├── orchestrator/       # LangGraph state-management logic
├── rag/                # PDF processing and vector retrieval
├── static/             # Assets (CSS, JS, Custom Avatars)
├── templates/          # HTML Templates
├── requirements.txt    # Project dependencies
└── app.py              # Application entry point
```

## 🚀 Installation & Setup

### 1. Prerequisite
Ensure you have Python 3.9+ and a [Groq API Key](https://console.groq.com/).

### 2. Implementation
```bash
# Clone the repository
git clone <your-repo-url>
cd nexus-ai

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Config
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL_NAME=llama-3.3-70b-versatile
```

### 4. Run the Application
```bash
python app.py
```
Visit `http://127.0.0.1:5000` to start chatting.

## 📄 License
This project is for educational and portfolio purposes.
