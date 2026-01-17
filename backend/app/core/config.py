"""
Application configuration.
"""

import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SEO Content Optimizer"

    # CORS - explicit origins (wildcards don't work with FastAPI CORS)
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://frontend-ke4wjrvr0-johan-cilliers-projects.vercel.app",
    ]

    # Allow all origins (set to True for debugging CORS issues)
    CORS_ALLOW_ALL: bool = True

    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".docx"]

    # Optimization defaults
    DEFAULT_MIN_KEYWORD_DENSITY: float = 1.0
    DEFAULT_MAX_KEYWORD_DENSITY: float = 2.5
    DEFAULT_FAQ_COUNT: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
