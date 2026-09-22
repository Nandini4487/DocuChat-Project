"""Unit and integration tests for RAG service PDF ingestion, chunking, vector storage, and QA."""

from pathlib import Path
from typing import List
import pytest

from backend.config import get_settings
from backend.rag_service import (
    NO_INFO_RESPONSE,
    EmptyDocumentError,
    RAGService,
    answer_question,
    compute_file_hash,
    get_chunks_by_metadata,
    load_and_chunk_pdf,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures"
SAMPLE_PDF_PATH = FIXTURES_DIR / "sample_2page.pdf"
EMPTY_PDF_PATH = FIXTURES_DIR / "empty_scanned.pdf"


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
    """Mock Chat LLM for deterministic, zero-network unit tests."""

    def __init__(self, canned_response: str = "This is a mock answer.") -> None:
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
# Task 3 Tests: PDF Ingestion & Chunking
# ---------------------------------------------------------------------------

def test_load_and_chunk_pdf_success():
    """Test that a 2-page PDF produces valid chunks with correct metadata and sizing."""
    settings = get_settings()
    custom_filename = "annual_report_2026.pdf"

    chunks = load_and_chunk_pdf(
        file_path=str(SAMPLE_PDF_PATH),
        original_filename=custom_filename,
    )

    # 1. Produces at least one chunk
    assert len(chunks) >= 1

    # 2. Every chunk's content length is <= CHUNK_SIZE
    for chunk in chunks:
        assert len(chunk.page_content) <= settings.CHUNK_SIZE
        assert len(chunk.page_content.strip()) > 0

    # 3. Page numbers are 1-indexed (1 and 2, never 0)
    page_numbers = {chunk.metadata["page"] for chunk in chunks}
    assert 0 not in page_numbers
    assert 1 in page_numbers
    assert 2 in page_numbers

    # 4. Chunk indices increase in sequential order (0, 1, ...)
    indices = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))

    # 5. Metadata source uses the original filename
    for chunk in chunks:
        assert chunk.metadata["source"] == custom_filename


def test_load_and_chunk_pdf_default_source():
    """Test that chunk metadata defaults to file basename when original_filename is omitted."""
    chunks = load_and_chunk_pdf(file_path=str(SAMPLE_PDF_PATH))
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.metadata["source"] == SAMPLE_PDF_PATH.name


def test_empty_or_scanned_pdf_chunking_error():
    """Test that chunking a PDF with no extractable text layer raises EmptyDocumentError."""
    with pytest.raises(EmptyDocumentError) as exc_info:
        load_and_chunk_pdf(file_path=str(EMPTY_PDF_PATH))

    assert "no extractable text" in str(exc_info.value).lower()


def test_rag_service_instance_chunking():
    """Test that RAGService class instance method wraps chunking appropriately."""
    service = RAGService(
        embedding_function=DeterministicFakeEmbeddings(),
        llm=MockChatLLM(),
    )
    chunks = service.load_and_chunk_pdf(file_path=str(SAMPLE_PDF_PATH))
    assert len(chunks) >= 1
    assert chunks[0].metadata["chunk_index"] == 0


# ---------------------------------------------------------------------------
# Task 4 Tests: Vector Store Ingestion & Chroma Persistence (No Network Calls)
# ---------------------------------------------------------------------------

def test_vector_store_ingest_and_query_by_metadata(tmp_path):
    """Test that ingesting sample PDF adds chunks to Chroma with correct metadata."""
    persist_dir = str(tmp_path / "chroma_test_db")
    fake_embeddings = DeterministicFakeEmbeddings()
    mock_llm = MockChatLLM()
    service = RAGService(
        embedding_function=fake_embeddings,
        persist_directory=persist_dir,
        llm=mock_llm,
    )

    sample_filename = "docuchat_report.pdf"
    expected_hash = compute_file_hash(SAMPLE_PDF_PATH.read_bytes())

    # Ingest document
    result = service.ingest_pdf(
        file_path=str(SAMPLE_PDF_PATH),
        original_filename=sample_filename,
    )

    # 1. Validate IngestionResult
    assert result.already_indexed is False
    assert result.chunks_created > 0
    assert result.file_hash == expected_hash
    assert result.filename == sample_filename

    # 2. Query vector store using metadata filter (source == sample_filename)
    stored_chunks = service.get_chunks_by_metadata({"source": sample_filename})
    assert len(stored_chunks) == result.chunks_created

    # 3. Assert chunk metadata integrity
    for chunk in stored_chunks:
        assert chunk.metadata["source"] == sample_filename
        assert chunk.metadata["file_hash"] == expected_hash
        assert chunk.metadata["page"] in (1, 2)
        assert len(chunk.page_content.strip()) > 0

    # 4. Test standalone helper get_chunks_by_metadata
    helper_chunks = get_chunks_by_metadata(service.vector_store, {"file_hash": expected_hash})
    assert len(helper_chunks) == result.chunks_created


def test_vector_store_duplicate_detection(tmp_path):
    """Test that ingesting identical file bytes twice is detected and skips re-embedding."""
    persist_dir = str(tmp_path / "chroma_test_dup_db")
    fake_embeddings = DeterministicFakeEmbeddings()
    mock_llm = MockChatLLM()
    service = RAGService(
        embedding_function=fake_embeddings,
        persist_directory=persist_dir,
        llm=mock_llm,
    )

    # First ingestion
    first_result = service.ingest_pdf(
        file_path=str(SAMPLE_PDF_PATH),
        original_filename="sample_first.pdf",
    )
    assert first_result.already_indexed is False
    assert first_result.chunks_created > 0

    initial_chunk_ids = service.vector_store.get()["ids"]
    initial_count = len(initial_chunk_ids)
    assert initial_count == first_result.chunks_created

    # Second ingestion of the exact same PDF bytes
    second_result = service.ingest_pdf(
        file_path=str(SAMPLE_PDF_PATH),
        original_filename="sample_second.pdf",
    )

    # Assert duplicate was detected
    assert second_result.already_indexed is True
    assert second_result.chunks_created == 0
    assert second_result.file_hash == first_result.file_hash

    # Assert collection count did NOT change
    post_dup_chunk_ids = service.vector_store.get()["ids"]
    assert len(post_dup_chunk_ids) == initial_count


# ---------------------------------------------------------------------------
# Task 5 Tests: Retrieval & Grounded Answer Generation (No Network Calls)
# ---------------------------------------------------------------------------

def test_answer_question_with_mocked_context(tmp_path):
    """Test that retrieved context chunks produce answer with correct deduplicated sources."""
    persist_dir = str(tmp_path / "chroma_qa_db")
    fake_embeddings = DeterministicFakeEmbeddings()
    mock_llm = MockChatLLM(canned_response="DocuChat is a RAG-powered document QA assistant.")
    service = RAGService(
        embedding_function=fake_embeddings,
        persist_directory=persist_dir,
        llm=mock_llm,
    )

    # Ingest the 2-page sample PDF
    service.ingest_pdf(str(SAMPLE_PDF_PATH), original_filename="sample_2page.pdf")

    # Run query
    result = service.answer_question("What is DocuChat?")

    # 1. Answer matches mocked LLM response
    assert result["answer"] == "DocuChat is a RAG-powered document QA assistant."
    assert mock_llm.call_count == 1

    # 2. Sources are properly populated and deduplicated
    assert len(result["sources"]) >= 1
    for src in result["sources"]:
        assert src["file"] == "sample_2page.pdf"
        assert src["page"] in (1, 2)

    # Also test top-level answer_question convenience function
    top_level_result = answer_question("What is DocuChat?", rag_service=service)
    assert top_level_result["answer"] == "DocuChat is a RAG-powered document QA assistant."


def test_answer_question_no_context_skips_llm(tmp_path):
    """Test that when vector store is empty, answer_question returns NO_INFO_RESPONSE without calling LLM."""
    persist_dir = str(tmp_path / "empty_chroma_qa_db")
    fake_embeddings = DeterministicFakeEmbeddings()
    mock_llm = MockChatLLM()
    service = RAGService(
        embedding_function=fake_embeddings,
        persist_directory=persist_dir,
        llm=mock_llm,
    )

    # Ask question on empty vector store
    result = service.answer_question("What is the quarterly revenue?")

    # 1. Exact refusal string returned
    assert result["answer"] == NO_INFO_RESPONSE
    assert result["answer"] == "I don't know based on the provided documents."

    # 2. Empty sources list
    assert result["sources"] == []

    # 3. Assert LLM was never called
    assert mock_llm.call_count == 0


@pytest.mark.integration
def test_answer_question_real_integration(tmp_path):
    """Live integration test using Google Gemini API and real embeddings/LLM.

    Run separately using:
        pytest backend/tests/test_rag_service.py -m integration -v -s
    """
    settings = get_settings()
    api_key = settings.GOOGLE_API_KEY.get_secret_value()
    if not api_key:
        pytest.skip("GOOGLE_API_KEY is not configured in .env; skipping live integration test.")

    persist_dir = str(tmp_path / "chroma_live_integration_db")
    service = RAGService(persist_directory=persist_dir)

    # Ingest sample PDF
    ingest_result = service.ingest_pdf(
        file_path=str(SAMPLE_PDF_PATH),
        original_filename="sample_2page.pdf",
    )
    assert ingest_result.already_indexed is False
    assert ingest_result.chunks_created > 0

    # Page 1 contains: "DocuChat Page 1: Retrieval Augmented Generation document ingestion test."
    response = service.answer_question("What is on Page 1?")

    assert len(response["answer"]) > 0
    assert response["answer"] != NO_INFO_RESPONSE
    # Assert response contains citation for Page 1
    cited_pages = [s["page"] for s in response["sources"] if s["file"] == "sample_2page.pdf"]
    assert 1 in cited_pages
