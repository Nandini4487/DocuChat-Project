"""Integration tests for FastAPI REST API endpoints using TestClient with mocked Gemini models."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import app
from backend.rag_service import EmptyDocumentError, IngestionResult, NO_INFO_RESPONSE


# ---------------------------------------------------------------------------
# GET /health Tests
# ---------------------------------------------------------------------------


def test_health_endpoint(client: TestClient):
    """1. GET /health returns 200 OK and healthy status payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["vector_store_ready"] is True


# ---------------------------------------------------------------------------
# POST /upload Tests
# ---------------------------------------------------------------------------


def test_upload_sample_pdf_success(client: TestClient, sample_pdf_path: Path, mock_rag_service: MagicMock):
    """2. POST /upload with sample_2page.pdf returns 200 with correct filename and chunk count."""
    pdf_bytes = sample_pdf_path.read_bytes()
    mock_rag_service.ingest_pdf.return_value = IngestionResult(
        already_indexed=False,
        chunks_created=2,
        file_hash="test_sha256_hash",
        filename="sample_2page.pdf",
        message="Document 'sample_2page.pdf' successfully indexed with 2 chunks.",
    )

    response = client.post(
        "/upload",
        files={"file": ("sample_2page.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample_2page.pdf"
    assert data["chunks_created"] == 2
    assert "successfully indexed" in data["message"]
    mock_rag_service.ingest_pdf.assert_called_once()


def test_upload_duplicate_detection_success(
    client: TestClient, sample_pdf_path: Path, mock_rag_service: MagicMock
):
    """3. POST /upload with the same file again is detected as duplicate and returns 200 without re-embedding."""
    pdf_bytes = sample_pdf_path.read_bytes()
    mock_rag_service.ingest_pdf.return_value = IngestionResult(
        already_indexed=True,
        chunks_created=0,
        file_hash="test_sha256_hash",
        filename="sample_2page.pdf",
        message="Document 'sample_2page.pdf' is already indexed.",
    )
    mock_rag_service.get_chunks_by_metadata.return_value = [
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
    ]

    response = client.post(
        "/upload",
        files={"file": ("sample_2page.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample_2page.pdf"
    assert data["chunks_created"] == 2
    assert "already indexed" in data["message"]


def test_upload_non_pdf_file_rejected(client: TestClient):
    """4. POST /upload with a .txt file returns HTTP 400 Bad Request."""
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "Only PDF files are supported."}


def test_upload_oversized_file_rejected(
    client: TestClient, mock_rag_service: MagicMock, tmp_path: Path
):
    """5. POST /upload with a file exceeding MAX_UPLOAD_MB returns HTTP 400."""
    custom_settings = Settings(
        UPLOAD_DIR=str(tmp_path / "uploads_limit"),
        MAX_UPLOAD_MB=1,
    )
    app.dependency_overrides[get_settings] = lambda: custom_settings

    oversized_data = b"%PDF-1.4 " + (b"0" * (1024 * 1024 + 100))
    response = client.post(
        "/upload",
        files={"file": ("huge.pdf", oversized_data, "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "File exceeds the 1 MB limit."}

    # Verify partially uploaded file was deleted
    saved_file = tmp_path / "uploads_limit" / "huge.pdf"
    assert not saved_file.exists()
    mock_rag_service.ingest_pdf.assert_not_called()


def test_upload_empty_scanned_pdf_rejected(
    client: TestClient, empty_pdf_path: Path, mock_rag_service: MagicMock
):
    """6. POST /upload with empty_scanned.pdf returns HTTP 400 with EmptyDocumentError message."""
    pdf_bytes = empty_pdf_path.read_bytes()
    error_msg = "No extractable text found in PDF 'empty_scanned.pdf'. The document may be scanned or image-only."
    mock_rag_service.ingest_pdf.side_effect = EmptyDocumentError(error_msg)

    response = client.post(
        "/upload",
        files={"file": ("empty_scanned.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": error_msg}


# ---------------------------------------------------------------------------
# POST /ask Tests
# ---------------------------------------------------------------------------


def test_ask_empty_question_rejected(client: TestClient):
    """7. POST /ask with an empty question returns HTTP 422 Unprocessable Entity."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_grounded_question_success(client: TestClient, mock_rag_service: MagicMock):
    """8. POST /ask with question answerable from document returns 200 with answer and source citations."""
    mock_rag_service.answer_question.return_value = {
        "answer": "DocuChat Page 1 tests document ingestion.",
        "sources": [{"file": "sample_2page.pdf", "page": 1}],
    }

    response = client.post("/ask", json={"question": "What is on page 1?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "DocuChat Page 1 tests document ingestion."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["file"] == "sample_2page.pdf"
    assert data["sources"][0]["page"] == 1


def test_ask_unrelated_question_refusal(client: TestClient, mock_rag_service: MagicMock):
    """9. POST /ask with unrelated question returns 200 with refusal string and empty sources."""
    mock_rag_service.answer_question.return_value = {
        "answer": NO_INFO_RESPONSE,
        "sources": [],
    }

    response = client.post("/ask", json={"question": "What is the recipe for chocolate cake?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "I don't know based on the provided documents."
    assert data["sources"] == []


def test_ask_llm_exception_returns_500(client: TestClient, mock_rag_service: MagicMock):
    """10. POST /ask when mocked LLM raises exception returns 500 with generic message."""
    mock_rag_service.answer_question.side_effect = Exception("Google GenAI 503 Service Unavailable")

    response = client.post("/ask", json={"question": "Any question?"})
    assert response.status_code == 500
    data = response.json()
    assert data == {"detail": "Something went wrong while generating the answer."}
    # Ensure internal exception details are not leaked to client
    assert "Google GenAI 503" not in response.text
