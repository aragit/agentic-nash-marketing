"""Pydantic Settings for environment-based configuration."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    database_url: str = Field(default="sqlite:///./nash_marketing.db")

    # LLM
    llm_backend: str = Field(default="mock")
    llm_model: str = Field(default="microsoft/Phi-3-mini-4k-instruct")

    # App
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()