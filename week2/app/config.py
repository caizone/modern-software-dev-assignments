"""
Centralized configuration management using Pydantic Settings.

Rationale:
- Single source of truth for all configuration values
- Environment variable support with type validation
- Testable configuration with easy overrides
- Proper defaults with documentation
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Database settings
    db_path: Path = Field(
        default=Path(__file__).resolve().parents[1] / "data" / "app.db",
        description="Path to SQLite database file",
    )

    # LLM settings
    llm_model: str = Field(
        default="codellama:7b",
        description="Ollama model to use for extraction",
    )
    llm_temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="LLM generation temperature (lower = more deterministic)",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Timeout for LLM API calls",
    )

    # Application settings
    app_name: str = Field(
        default="Action Item Extractor",
        description="Application name for FastAPI docs",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode (verbose logging)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Using lru_cache ensures we only parse env vars once,
    and subsequent calls return the same instance.
    """
    return Settings()
