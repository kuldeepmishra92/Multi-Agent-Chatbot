from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config
from utils.logger import get_logger

logger = get_logger(__name__)

def load_pdf(file_path: str) -> List[Document]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    logger.info(f"Loading PDF: {path.name}")
    loader = PyPDFLoader(str(path))
    pages = loader.load()
    logger.info(f"  → {len(pages)} pages loaded from '{path.name}'")
    return pages

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"  → {len(chunks)} chunks created "
                f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks

def load_and_chunk_pdf(file_path: str) -> List[Document]:
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    logger.info(f"PDF pipeline complete: {len(chunks)} chunks ready for indexing.")
    return chunks
