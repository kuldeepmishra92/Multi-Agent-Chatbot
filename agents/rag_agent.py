from typing import List, Dict
from langchain_core.documents import Document
from groq import Groq
from agents.base_agent import BaseAgent
from rag.retriever import hybrid_search
from rag.vector_store import get_document_count
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_MIN_CHUNK_LENGTH = 30

def _format_context(chunks: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page   = doc.metadata.get("page", "?")
        parts.append(
            f"[Chunk {i}] Source: {source} | Page: {page}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)

def _format_sources(chunks: List[Document]) -> str:
    seen, sources = set(), []
    for doc in chunks:
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key  = f"{src}:p{page}"
        if key not in seen:
            seen.add(key)
            sources.append(f"📄 `{src}` (page {page})")
    return "\n".join(sources) if sources else ""

def _chunks_are_relevant(chunks: List[Document]) -> bool:
    meaningful = [c for c in chunks if len(c.page_content.strip()) >= _MIN_CHUNK_LENGTH]
    return len(meaningful) > 0

class RAGAgent(BaseAgent):
    RAG_SYSTEM_PROMPT = """You are a precise document-based question answering assistant.
Answer the user's question using ONLY the provided document context below.

Rules:
- Base your answer strictly on the provided context — do NOT use outside knowledge.
- Quote or paraphrase from the context where helpful.
- Be concise and direct.
- At the end, briefly mention which chunk(s) you used.
- Format with markdown. IMPORTANT: Always include a space after bolding (e.g., **Source:** text) for correct UI rendering.

Document Context:
{context}"""

    GENERAL_SYSTEM_PROMPT = """You are a knowledgeable, helpful AI assistant.
Answer the user's question clearly and accurately using your training knowledge.
Be concise but thorough. Format with markdown. Always include a space after bolding (e.g., **Note:** text) for correct UI rendering."""

    def __init__(self):
        super().__init__(name="RAG Agent")
        self._client = Groq(api_key=config.GROQ_API_KEY)
        logger.info("RAGAgent (with general fallback) ready.")

    def run(
        self,
        query: str,
        context: str = "",
        history: List[Dict[str, str]] = None,
        session_id: str = "",
    ) -> str:
        logger.info(f"RAGAgent processing: '{query[:80]}'")

        doc_count = 0
        try:
            doc_count = get_document_count()
        except Exception:
            pass

        chunks = []
        if doc_count > 0:
            logger.info(f"  → Running hybrid retrieval over {doc_count} chunks...")
            chunks = hybrid_search(query, k=config.TOP_K_RETRIEVAL)

        if chunks and _chunks_are_relevant(chunks):
            logger.info(f"  → {len(chunks)} relevant chunks found. Using RAG mode.")
            return self._rag_answer(query, chunks, history)
        else:
            reason = "no documents indexed" if doc_count == 0 else "no relevant chunks found in documents"
            logger.info(f"  → {reason}. Falling back to general knowledge.")
            return self._general_answer(query, history, fallback_reason=reason)

    def _rag_answer(
        self,
        query: str,
        chunks: List[Document],
        history: List[Dict[str, str]],
    ) -> str:
        context_block = _format_context(chunks)
        sources_block = _format_sources(chunks)

        messages = [{"role": "system",
                     "content": self.RAG_SYSTEM_PROMPT.format(context=context_block)}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": query})

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=config.GROQ_TEMPERATURE,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()
            if sources_block:
                answer += f"\n\n---\n**Sources used:**\n{sources_block}"
            logger.info("  → RAG answer generated.")
            return answer
        except Exception as exc:
            logger.error(f"RAGAgent (RAG path) failed: {exc}")
            return self._general_answer(query, history, fallback_reason="LLM error")

    def _general_answer(
        self,
        query: str,
        history: List[Dict[str, str]],
        fallback_reason: str = "",
    ) -> str:
        messages = [{"role": "system", "content": self.GENERAL_SYSTEM_PROMPT}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": query})

        try:
            response = self._client.chat.completions.create(
                model=config.GROQ_MODEL_NAME,
                messages=messages,
                temperature=0.5,
                max_tokens=config.GROQ_MAX_TOKENS,
            )
            answer = response.choices[0].message.content.strip()

            answer += (
                "\n\n---\n"
                "*💡 This answer is from general knowledge — "
                "no relevant content was found in your uploaded documents.*"
            )
            logger.info(f"  → General fallback answer generated ({fallback_reason}).")
            return answer

        except Exception as exc:
            logger.error(f"RAGAgent (general path) failed: {exc}")
            return f"I encountered an error answering your question. Please try again.\n\nError: {exc}"
