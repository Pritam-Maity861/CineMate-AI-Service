from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    PORT: int = 8000
    ENVIRONMENT: str = "development"
    CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]
    FRONTEND_URL: str = "http://localhost:5173"

    # Node.js Core Backend URL
    NODE_API_URL: str = "http://localhost:8080/api/v1"

    # PostgreSQL with pgvector
    DATABASE_URL: Optional[str] = None

    # Groq API Configuration (Default / Recommended)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    # Other LLM Providers (Optional fallbacks)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Default Model Provider
    DEFAULT_MODEL_PROVIDER: str = "groq"  # "groq", "gemini", or "openai"
    EMBEDDING_MODEL: str = "models/text-embedding-004"


settings = Settings()

