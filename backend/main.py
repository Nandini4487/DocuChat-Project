"""DocuChat FastAPI application backend entrypoint and routing."""

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader

from backend.config import Settings, get_settings
from backend.rag_service import EmptyDocumentError, RAGService
from backend.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceCitation,
    UploadResponse,
)

logger = logging.getLogger(__name__)

# Schema aliases for specification compatibility
AskRequest = QueryRequest
AskResponse = QueryResponse

# Hardcoded origins for local Streamlit and frontend development
ALLOWED_ORIGINS = [
    "http://localhost:8501",
    "http://localhost:3000",
]

app = FastAPI(
    title="DocuChat API",
    version="1.0.0",
    description="RAG-powered document question answering backend API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_rag_service() -> RAGService:
    """Dependency provider for RAGService instance."""
    return RAGService()


@app.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Health check endpoint returning service status and readiness."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        vector_store_ready=True,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    rag_service: RAGService = Depends(get_rag_service),
) -> UploadResponse:
    """Upload a PDF file, save it to UPLOAD_DIR, and index chunks into Chroma."""
    # Validate file type (extension and MIME content-type)
    if not (file.filename and file.filename.lower().endswith(".pdf")) or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported.",
        )

    upload_dir = Path(settings.UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)

    filename = file.filename or "uploaded_document.pdf"
    dest_path = upload_dir / filename

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    total_bytes = 0
    chunk_size = 1024 * 1024  # 1 MB chunk

    # Stream chunks and check file size limit
    try:
        with open(dest_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    f.close()
                    if dest_path.exists():
                        dest_path.unlink()
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File exceeds the {settings.MAX_UPLOAD_MB} MB limit.",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()
        logger.exception("Error saving uploaded file: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while processing the file.",
        )

    # Ingest document into Chroma vector database
    try:
        result = rag_service.ingest_pdf(
            file_path=str(dest_path),
            original_filename=filename,
        )
    except EmptyDocumentError as e:
        logger.warning("Empty or scanned PDF uploaded (%s): %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error during PDF ingestion for '%s': %s", filename, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while processing the file.",
        )

    # Calculate total pages from the uploaded PDF
    try:
        reader = PdfReader(str(dest_path))
        total_pages = len(reader.pages)
    except Exception:
        total_pages = 1

    # If already indexed (duplicate), fetch existing chunk count and return 200 OK
    if result.already_indexed:
        existing_chunks = rag_service.get_chunks_by_metadata({"file_hash": result.file_hash})
        chunks_count = len(existing_chunks)
        status_str = "success"
    else:
        chunks_count = result.chunks_created
        status_str = "success"

    return UploadResponse(
        status=status_str,
        filename=filename,
        total_pages=total_pages,
        chunks_created=chunks_count,
        message=result.message,
    )


@app.post("/ask", response_model=QueryResponse)
def ask_question(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer questions grounded on indexed PDF documents with source citations."""
    try:
        result = rag_service.answer_question(request.question)
        sources = [SourceCitation(**src) for src in result.get("sources", [])]
        return QueryResponse(
            answer=result.get("answer", ""),
            sources=sources,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during question answering: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while generating the answer.",
        )

