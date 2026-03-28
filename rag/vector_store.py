from typing import List, Optional
import sys

# Support for Streamlit Cloud / Linux environments with old SQLite
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

from langchain_core.documents import Document
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import config
from utils.logger import get_logger

logger = get_logger(__name__)

_embedding_model: Optional[HuggingFaceEmbeddings] = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        _embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("  → Embedding model loaded successfully.")
    return _embedding_model

_vector_store: Optional[Chroma] = None

def get_vector_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        logger.info(f"Connecting to ChromaDB at: {config.CHROMA_DB_PATH}")
        _vector_store = Chroma(
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=get_embedding_model(),
            persist_directory=config.CHROMA_DB_PATH,
        )
        count = _vector_store._collection.count()
        logger.info(f"  → ChromaDB ready. Documents in store: {count}")
    return _vector_store

def add_documents(documents: List[Document]) -> int:
    if not documents:
        logger.warning("add_documents called with empty list — nothing to index.")
        return 0

    store = get_vector_store()
    logger.info(f"Indexing {len(documents)} chunks into ChromaDB...")
    store.add_documents(documents)
    logger.info(f"  → Indexing complete. Total in store: {store._collection.count()}")
    return len(documents)

def similarity_search(query: str, k: int = None) -> List[Document]:
    k = k or config.TOP_K_RETRIEVAL
    store = get_vector_store()

    logger.info(f"Semantic search (k={k}): '{query[:60]}...' " if len(query) > 60
                else f"Semantic search (k={k}): '{query}'")

    results = store.similarity_search(query, k=k)
    logger.info(f"  → {len(results)} chunks retrieved.")
    return results

def get_document_count() -> int:
    return get_vector_store()._collection.count()

def clear_store() -> None:
    global _vector_store
    store = get_vector_store()
    store.delete_collection()
    _vector_store = None
    logger.warning("ChromaDB collection cleared.")
