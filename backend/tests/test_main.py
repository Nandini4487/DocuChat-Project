"""Tests for FastAPI backend application routes, validation, and error handling."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import app, get_rag_service
from backend.rag_service import EmptyDocumentError, IngestionResult


@pytest.fixture
def mock_rag_service():
    """Mock RAGService fixture ensuring zero network calls."""
    mock = MagicMock()
    return mock


@pytest.fixture
def client(mock_rag_service, tmp_path):
    """FastAPI TestClient with overridden dependencies for testing."""
    test_settings = Settings(
        UPLOAD_DIR=str(tmp_path / "uploads"),
        CHROMA_DIR=str(tmp_path / "chroma_db"),
        MAX_UPLOAD_MB=20,
    )

    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# GET /health Tests
# ---------------------------------------------------------------------------


def test_health_endpoint(client):
    """Verify GET /health returns 200 OK with the expected health status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert data["vector_store_ready"] is True


# ---------------------------------------------------------------------------
# POST /upload Tests
# ---------------------------------------------------------------------------


def test_upload_non_pdf_extension_rejected(client):
    """Uploading a .txt file is rejected with HTTP 400 and clear error message."""
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"Some plain text content", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "Only PDF files are supported."}


def test_upload_wrong_mime_type_rejected(client):
    """Uploading a file with non-PDF MIME type is rejected with HTTP 400."""
    response = client.post(
        "/upload",
        files={"file": ("document.pdf", b"Fake PDF content", "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "Only PDF files are supported."}


def test_upload_non_pdf_extension_with_pdf_mime_rejected(client):
    """Uploading a non-PDF extension with PDF MIME type is rejected with HTTP 400."""
    response = client.post(
        "/upload",
        files={"file": ("image.png", b"Fake content", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "Only PDF files are supported."}


def test_upload_oversized_file_rejected(client, mock_rag_service, tmp_path):
    """Uploading a file exceeding MAX_UPLOAD_MB is rejected with HTTP 400 and deleted."""
    # Override settings with a 1 MB limit for fast, deterministic test
    custom_settings = Settings(
        UPLOAD_DIR=str(tmp_path / "uploads_small"),
        MAX_UPLOAD_MB=1,
    )
    app.dependency_overrides[get_settings] = lambda: custom_settings

    oversized_content = b"%PDF-1.4 " + (b"0" * (1024 * 1024 + 500))  # ~1.0005 MB

    response = client.post(
        "/upload",
        files={"file": ("large_document.pdf", oversized_content, "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": "File exceeds the 1 MB limit."}

    # Verify no file is left behind in the upload directory
    upload_dir = tmp_path / "uploads_small"
    saved_file = upload_dir / "large_document.pdf"
    assert not saved_file.exists()

    # Verify rag_service was never called for ingestion
    mock_rag_service.ingest_pdf.assert_not_called()


def test_upload_empty_document_error_handling(client, mock_rag_service):
    """Catch EmptyDocumentError from rag_service and return HTTP 400 with its message."""
    error_message = (
        "No extractable text found in PDF 'scanned.pdf'. "
        "The document may be scanned, image-only, or password-protected."
    )
    mock_rag_service.ingest_pdf.side_effect = EmptyDocumentError(error_message)

    fake_pdf = b"%PDF-1.4 fake content"
    response = client.post(
        "/upload",
        files={"file": ("scanned.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data == {"detail": error_message}


def test_upload_unexpected_exception_returns_500(client, mock_rag_service):
    """Catch unexpected exception from rag_service and return generic HTTP 500 message."""
    mock_rag_service.ingest_pdf.side_effect = RuntimeError("Internal Chroma SQLite locking error")

    fake_pdf = b"%PDF-1.4 fake content"
    response = client.post(
        "/upload",
        files={"file": ("broken.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 500
    data = response.json()
    assert data == {"detail": "Something went wrong while processing the file."}
    # Ensure internal exception details are not leaked
    assert "Internal Chroma SQLite locking error" not in response.text


def test_upload_success_returns_200(client, mock_rag_service):
    """Successful PDF upload returns HTTP 200 with UploadResponse schema."""
    mock_rag_service.ingest_pdf.return_value = IngestionResult(
        already_indexed=False,
        chunks_created=12,
        file_hash="mocked_sha256_hash",
        filename="sample.pdf",
        message="Document successfully indexed (12 chunks created).",
    )

    fake_pdf = b"%PDF-1.4 fake content"
    response = client.post(
        "/upload",
        files={"file": ("sample.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "sample.pdf"
    assert data["chunks_created"] == 12
    assert "Document successfully indexed" in data["message"]


def test_upload_duplicate_already_indexed_returns_200(client, mock_rag_service):
    """Duplicate PDF upload returns HTTP 200 with existing chunk count."""
    mock_rag_service.ingest_pdf.return_value = IngestionResult(
        already_indexed=True,
        chunks_created=0,
        file_hash="mocked_duplicate_hash",
        filename="duplicate.pdf",
        message="Document already indexed. Skipping re-embedding.",
    )
    mock_rag_service.get_chunks_by_metadata.return_value = [
        {"chunk_id": "c1"},
        {"chunk_id": "c2"},
        {"chunk_id": "c3"},
    ]

    fake_pdf = b"%PDF-1.4 fake content"
    response = client.post(
        "/upload",
        files={"file": ("duplicate.pdf", fake_pdf, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "duplicate.pdf"
    assert data["chunks_created"] == 3
    assert "Document already indexed" in data["message"]


# ---------------------------------------------------------------------------
# POST /ask Tests
# ---------------------------------------------------------------------------


def test_ask_empty_question_rejected(client):
    """POST /ask with empty string question is rejected with HTTP 422 by Pydantic."""
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_missing_question_field_rejected(client):
    """POST /ask with missing question field is rejected with HTTP 422 by Pydantic."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_unexpected_exception_returns_500(client, mock_rag_service):
    """Catch unexpected exception from rag_service during QA and return generic HTTP 500."""
    mock_rag_service.answer_question.side_effect = Exception("Google GenAI 503 Service Unavailable")

    response = client.post("/ask", json={"question": "What is the revenue growth?"})
    assert response.status_code == 500
    data = response.json()
    assert data == {"detail": "Something went wrong while generating the answer."}
    # Ensure internal exception details are not leaked
    assert "Google GenAI 503" not in response.text


def test_ask_success_returns_200(client, mock_rag_service):
    """Successful POST /ask returns HTTP 200 with QueryResponse schema."""
    mock_rag_service.answer_question.return_value = {
        "answer": "Revenue grew by 14.5% year-over-year.",
        "sources": [{"file": "report.pdf", "page": 4}],
    }

    response = client.post("/ask", json={"question": "What was the revenue growth?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Revenue grew by 14.5% year-over-year."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["file"] == "report.pdf"
    assert data["sources"][0]["page"] == 4
