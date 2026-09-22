"""Configuration settings for DocuChat backend using pydantic-settings."""

from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings and environment variable schema."""

    # Google Gemini API Configuration
    GOOGLE_API_KEY: SecretStr = SecretStr("")

    # Model Settings
    LLM_MODEL: str = "gemini-3.6-flash"
    EMBEDDING_MODEL: str = "gemini-embedding-001"

    # RAG Ingestion & Retrieval Hyperparameters
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4
    MAX_UPLOAD_MB: int = 20

    # Storage Paths
    UPLOAD_DIR: str = "uploads"
    CHROMA_DIR: str = "chroma_db"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached instance of application settings."""
    return Settings()
