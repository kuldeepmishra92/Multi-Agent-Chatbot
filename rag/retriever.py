from typing import List, Tuple
from collections import defaultdict
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from rag.vector_store import similarity_search, get_vector_store
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_bm25_index: BM25Okapi | None = None
_bm25_docs: List[Document] = []

def _build_bm25_index() -> None:
    global _bm25_index, _bm25_docs

    store = get_vector_store()
    total = store._collection.count()

    if total == 0:
        logger.warning("BM25 index skipped — no documents in ChromaDB yet.")
        _bm25_index = None
        _bm25_docs = []
        return

    logger.info(f"Building BM25 index over {total} chunks...")

    raw = store._collection.get(include=["documents", "metadatas"])
    _bm25_docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    tokenised_corpus = [doc.page_content.lower().split() for doc in _bm25_docs]
    _bm25_index = BM25Okapi(tokenised_corpus)
    logger.info(f"  → BM25 index built over {len(_bm25_docs)} chunks.")

def _bm25_search(query: str, k: int) -> List[Tuple[Document, float]]:
    global _bm25_index, _bm25_docs

    _build_bm25_index()

    if _bm25_index is None:
        return []

    tokens = query.lower().split()
    raw_scores = _bm25_index.get_scores(tokens)

    scored = sorted(
        zip(_bm25_docs, raw_scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top_score = scored[0][1] if scored and scored[0][1] > 0 else 1.0
    normalised = [
        (doc, score / top_score)
        for doc, score in scored[:k]
        if score > 0
    ]

    logger.info(f"  → BM25 returned {len(normalised)} results.")
    return normalised

def hybrid_search(query: str, k: int = None) -> List[Document]:
    k = k or config.TOP_K_RETRIEVAL
    logger.info(f"Hybrid retrieval (k={k}) for: '{query[:70]}'")

    semantic_results: List[Document] = similarity_search(query, k=k)

    bm25_results: List[Tuple[Document, float]] = _bm25_search(query, k=k)
    bm25_docs = [doc for doc, _ in bm25_results]

    RRF_K = 60
    scores: dict = defaultdict(float)
    doc_map: dict = {}

    for rank, doc in enumerate(semantic_results):
        key = doc.page_content[:120]
        scores[key] += 1.0 / (rank + 1 + RRF_K)
        doc_map[key] = doc

    for rank, doc in enumerate(bm25_docs):
        key = doc.page_content[:120]
        scores[key] += 1.0 / (rank + 1 + RRF_K)
        doc_map[key] = doc

    ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    final_results = [doc_map[key] for key in ranked_keys[:k]]

    logger.info(f"  → Hybrid fusion complete. Returning {len(final_results)} chunks.")
    return final_results

def invalidate_bm25_cache() -> None:
    global _bm25_index, _bm25_docs
    _bm25_index = None
    _bm25_docs = []
    logger.info("BM25 cache invalidated — will rebuild on next search.")
