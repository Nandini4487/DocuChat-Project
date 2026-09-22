"""Pydantic data models and schemas for DocuChat API contracts."""

from typing import List
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = "healthy"
    version: str = "1.0.0"
    vector_store_ready: bool = True


class UploadResponse(BaseModel):
    """Document upload and indexing response schema."""
    status: str
    filename: str
    total_pages: int
    chunks_created: int
    message: str


class SourceCitation(BaseModel):
    """Citation of a specific file and page number."""
    file: str
    page: int


class QueryRequest(BaseModel):
    """Question answering query request schema."""
    question: str = Field(..., min_length=1, description="Non-empty question text")


class QueryResponse(BaseModel):
    """Question answering query response schema."""
    answer: str
    sources: List[SourceCitation] = []
