"""RAG service engine for PDF ingestion, chunking, vector storage, retrieval, and grounded Q&A."""

import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Fixed refusal constant used across grounded QA generation and tests
NO_INFO_RESPONSE: str = "I don't know based on the provided documents."

RAG_PROMPT_TEMPLATE = """You are a helpful, accurate AI assistant for question answering over documents.
Answer the question strictly and ONLY using the provided context below.
If the context does not contain the answer, or if you cannot answer based solely on the context, reply with EXACTLY:
{refusal_response}
Do not make assumptions, extrapolate, or use outside knowledge.

Context:
{context}

Question:
{question}

Answer:"""


class EmptyDocumentError(Exception):
    """Raised when a PDF contains no extractable text (e.g. scanned image-only PDF)."""
    pass


class IngestionResult(BaseModel):
    """Result object returned after PDF ingestion and indexing."""
    already_indexed: bool
    chunks_created: int
    file_hash: str
    filename: str
    message: str = ""


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of file content bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def load_and_chunk_pdf(
    file_path: str,
    original_filename: Optional[str] = None,
    file_hash: Optional[str] = None,
) -> List[Document]:
    """Load a PDF file, extract text, validate content, and split into chunks.

    Args:
        file_path: Local path to the PDF file.
        original_filename: Optional original filename for source attribution metadata.
        file_hash: Optional SHA-256 hash of file bytes to include in metadata.

    Returns:
        List of LangChain Document chunk objects with normalized metadata.

    Raises:
        EmptyDocumentError: If the document contains no extractable text.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # Reject PDFs with no extractable text layer (e.g., scanned/image-only)
    combined_text = "".join(page.page_content for page in pages).strip()
    if not combined_text:
        raise EmptyDocumentError(
            "The PDF contains no extractable text. It may be a scanned document "
            "or image-only PDF without a selectable text layer."
        )

    # Determine display source name
    source_name = original_filename or Path(file_path).name

    # Normalize page numbers to 1-indexed and assign source name
    for page_doc in pages:
        raw_page = page_doc.metadata.get("page", 0)
        page_doc.metadata["page"] = int(raw_page) + 1
        page_doc.metadata["source"] = source_name
        if file_hash:
            page_doc.metadata["file_hash"] = file_hash

    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(pages)

    # Assign 0-indexed sequential chunk identifier across document and ensure metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        if file_hash:
            chunk.metadata["file_hash"] = file_hash

    return chunks


def get_chunks_by_metadata(
    vector_store: Chroma,
    filter_dict: Dict[str, Any],
) -> List[Document]:
    """Fetch stored chunks from a Chroma collection matching a metadata filter.

    Args:
        vector_store: Chroma vector store instance.
        filter_dict: Metadata filter dictionary (e.g. {"source": "sample.pdf"}).

    Returns:
        List of LangChain Document instances containing page_content and metadata.
    """
    results = vector_store.get(where=filter_dict)
    documents: List[Document] = []
    if not results:
        return documents

    doc_texts = results.get("documents") or []
    metadatas = results.get("metadatas") or []

    for text, meta in zip(doc_texts, metadatas):
        documents.append(Document(page_content=text, metadata=meta or {}))

    return documents


def retrieve_chunks(
    vector_store: Chroma,
    question: str,
    top_k: int = 4,
) -> List[Document]:
    """Retrieve top-k most relevant document chunks for a question.

    Args:
        vector_store: Chroma vector store instance.
        question: User query string.
        top_k: Number of relevant document chunks to return.

    Returns:
        List of top-k retrieved LangChain Document chunks.
    """
    return vector_store.similarity_search(query=question, k=top_k)


class RAGService:
    """Core RAG engine managing document ingestion, chunking, storage, retrieval, and QA."""

    def __init__(
        self,
        embedding_function: Optional[Any] = None,
        vector_store: Optional[Chroma] = None,
        llm: Optional[Any] = None,
        persist_directory: Optional[str] = None,
        collection_name: str = "documents",
    ) -> None:
        self.settings = get_settings()

        # 1. Initialize Embeddings model (GoogleGenerativeAIEmbeddings) if not injected
        if embedding_function is not None:
            self.embedding_fn = embedding_function
        else:
            api_key = self.settings.GOOGLE_API_KEY.get_secret_value()
            self.embedding_fn = GoogleGenerativeAIEmbeddings(
                model=self.settings.EMBEDDING_MODEL,
                google_api_key=api_key or None,
            )

        # 2. Initialize persistent Chroma vector store if not injected
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            chroma_dir = persist_directory or self.settings.CHROMA_DIR
            self.vector_store = Chroma(
                collection_name=collection_name,
                persist_directory=chroma_dir,
                embedding_function=self.embedding_fn,
            )

        # 3. Initialize LLM model (ChatGoogleGenerativeAI) if not injected
        if llm is not None:
            self.llm = llm
        else:
            api_key = self.settings.GOOGLE_API_KEY.get_secret_value() or "dummy_key"
            try:
                self.llm = ChatGoogleGenerativeAI(
                    model=self.settings.LLM_MODEL,
                    temperature=0.0,
                    api_key=api_key,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize ChatGoogleGenerativeAI with temperature=0.0 (%s). Falling back without temperature.",
                    e,
                )
                self.llm = ChatGoogleGenerativeAI(
                    model=self.settings.LLM_MODEL,
                    api_key=api_key,
                )

    def load_and_chunk_pdf(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
        file_hash: Optional[str] = None,
    ) -> List[Document]:
        """Delegate to load_and_chunk_pdf function."""
        return load_and_chunk_pdf(file_path, original_filename, file_hash)

    def ingest_pdf(
        self,
        file_path: str,
        original_filename: Optional[str] = None,
    ) -> IngestionResult:
        """Ingest a PDF file into Chroma vector store with SHA-256 duplicate detection.

        Args:
            file_path: Local filesystem path to the PDF file.
            original_filename: Optional display name for source attribution metadata.

        Returns:
            IngestionResult with already_indexed flag, chunks count, file_hash, and filename.
        """
        path = Path(file_path)
        source_name = original_filename or path.name

        # 1. Compute SHA-256 hash of the uploaded file bytes
        file_bytes = path.read_bytes()
        file_hash = compute_file_hash(file_bytes)

        # 2. Duplicate Detection: Query Chroma collection for existing chunk with file_hash
        existing = self.vector_store.get(where={"file_hash": file_hash})
        if existing and existing.get("ids") and len(existing["ids"]) > 0:
            return IngestionResult(
                already_indexed=True,
                chunks_created=0,
                file_hash=file_hash,
                filename=source_name,
                message=f"Document '{source_name}' is already indexed.",
            )

        # 3. Load and chunk PDF
        chunks = self.load_and_chunk_pdf(
            file_path=file_path,
            original_filename=source_name,
            file_hash=file_hash,
        )

        # Ensure all chunks carry file_hash
        for chunk in chunks:
            chunk.metadata["file_hash"] = file_hash

        # 4. Add chunks to Chroma vector store (embeds chunks)
        if chunks:
            self.vector_store.add_documents(chunks)

        return IngestionResult(
            already_indexed=False,
            chunks_created=len(chunks),
            file_hash=file_hash,
            filename=source_name,
            message=f"Document '{source_name}' successfully indexed with {len(chunks)} chunks.",
        )

    def get_chunks_by_metadata(
        self,
        filter_dict: Dict[str, Any],
    ) -> List[Document]:
        """Fetch stored chunks matching metadata filter."""
        return get_chunks_by_metadata(self.vector_store, filter_dict)

    def retrieve_chunks(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """Retrieve top matching document chunks from Chroma."""
        k = top_k or self.settings.TOP_K
        return retrieve_chunks(self.vector_store, question, top_k=k)

    def _invoke_llm(self, prompt_text: str) -> str:
        """Invoke Chat LLM with parameter error fallback."""
        try:
            response = self.llm.invoke(prompt_text)
            if hasattr(response, "content"):
                return str(response.content).strip()
            return str(response).strip()
        except Exception as e:
            err_msg = str(e).lower()
            if "temperature" in err_msg or "sampling" in err_msg or "parameter" in err_msg or "deprecated" in err_msg:
                logger.warning(
                    "LLM invocation failed with sampling parameter error (%s); falling back without temperature.",
                    e,
                )
                api_key = self.settings.GOOGLE_API_KEY.get_secret_value() or "dummy_key"
                fallback_llm = ChatGoogleGenerativeAI(
                    model=self.settings.LLM_MODEL,
                    api_key=api_key,
                )
                self.llm = fallback_llm
                response = self.llm.invoke(prompt_text)
                if hasattr(response, "content"):
                    return str(response.content).strip()
                return str(response).strip()
            raise

    def generate_grounded_answer(
        self,
        question: str,
        context_chunks: List[Document],
    ) -> Dict[str, Any]:
        """Generate a grounded answer and deduplicated source citations from retrieved chunks.

        Args:
            question: User query string.
            context_chunks: Retrieved relevant document chunks.

        Returns:
            Dict containing 'answer' string and 'sources' list of dicts.
        """
        if not context_chunks:
            return {
                "answer": NO_INFO_RESPONSE,
                "sources": [],
            }

        # Format context blocks with source and page markers
        context_parts = []
        for chunk in context_chunks:
            src = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", 1)
            context_parts.append(f"[Source: {src}, Page: {page}]\n{chunk.page_content}")
        context_str = "\n\n".join(context_parts)

        prompt_text = RAG_PROMPT_TEMPLATE.format(
            refusal_response=NO_INFO_RESPONSE,
            context=context_str,
            question=question,
        )

        answer_text = self._invoke_llm(prompt_text)

        # Deduplicate sources while preserving chunk order
        seen = set()
        sources: List[Dict[str, Any]] = []
        for chunk in context_chunks:
            src = chunk.metadata.get("source", "unknown")
            page = int(chunk.metadata.get("page", 1))
            key = (src, page)
            if key not in seen:
                seen.add(key)
                sources.append({"file": src, "page": page})

        return {
            "answer": answer_text,
            "sources": sources,
        }

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve context and generate answer with source citations.

        If no chunks are retrieved, returns NO_INFO_RESPONSE with empty sources
        without calling the LLM.

        Args:
            question: User query string.
            top_k: Optional top-k chunk count override.

        Returns:
            Dict with 'answer' and 'sources'.
        """
        k = top_k or self.settings.TOP_K
        chunks = self.retrieve_chunks(question, top_k=k)
        if not chunks:
            return {
                "answer": NO_INFO_RESPONSE,
                "sources": [],
            }
        return self.generate_grounded_answer(question, chunks)


def answer_question(
    question: str,
    rag_service: Optional[RAGService] = None,
) -> Dict[str, Any]:
    """Top-level convenience function for question answering.

    Args:
        question: User query string.
        rag_service: Optional RAGService instance.

    Returns:
        Dict with 'answer' and 'sources'.
    """
    service = rag_service or RAGService()
    return service.answer_question(question)
