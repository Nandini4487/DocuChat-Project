"""Shared pytest fixtures and test configuration for DocuChat."""

from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.config import Settings, get_settings
from backend.main import app, get_rag_service
from backend.rag_service import IngestionResult, RAGService

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SAMPLE_PDF_PATH = FIXTURES_DIR / "sample_2page.pdf"
EMPTY_PDF_PATH = FIXTURES_DIR / "empty_scanned.pdf"


# ---------------------------------------------------------------------------
# Deterministic Mock Embeddings & Chat LLM
# ---------------------------------------------------------------------------


class DeterministicFakeEmbeddings:
    """Deterministic mock embedding function requiring zero network calls or API keys."""

    def __init__(self, size: int = 768) -> None:
        self.size = size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            [float(((hash(t) + i) & 0x7FFFFFFF) % 1000) / 1000.0 for i in range(self.size)]
            for t in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return [
            [float(((hash(text) + i) & 0x7FFFFFFF) % 1000) / 1000.0 for i in range(self.size)][0]
        ] * self.size


class MockChatLLM:
    """Mock Chat LLM for deterministic, zero-network unit and integration tests."""

    def __init__(self, canned_response: str = "Mock answer from AI assistant.") -> None:
        self.canned_response = canned_response
        self.call_count = 0
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.call_count += 1
        self.last_prompt = prompt

        class MockAIMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        return MockAIMessage(self.canned_response)


# ---------------------------------------------------------------------------
# File Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_pdf_path() -> Path:
    """Path to sample 2-page text PDF fixture."""
    return SAMPLE_PDF_PATH


@pytest.fixture(scope="session")
def empty_pdf_path() -> Path:
    """Path to empty/scanned PDF fixture."""
    return EMPTY_PDF_PATH


# ---------------------------------------------------------------------------
# Isolated Settings & TestClient Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Isolated Settings instance pointing to temporary directories."""
    return Settings(
        UPLOAD_DIR=str(tmp_path / "uploads"),
        CHROMA_DIR=str(tmp_path / "chroma_db"),
        MAX_UPLOAD_MB=20,
    )


@pytest.fixture
def mock_rag_service() -> MagicMock:
    """Mock RAGService fixture ensuring zero network calls."""
    mock = MagicMock()
    return mock


@pytest.fixture
def client(mock_rag_service: MagicMock, test_settings: Settings) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with overridden dependencies for testing."""
    app.dependency_overrides[get_rag_service] = lambda: mock_rag_service
    app.dependency_overrides[get_settings] = lambda: test_settings

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
