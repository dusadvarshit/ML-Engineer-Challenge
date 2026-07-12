"""API Configuration Module"""
import os
from pathlib import Path
from typing import List


class Settings:
    # API Configuration
    API_TITLE: str = "ML Engineer Challenge API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/mlchallenge")

    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Model Configuration
    MODEL_CACHE_TTL: int = 3600  # 1 hour
    MAX_BATCH_SIZE: int = 32
    YOLO_MODEL_DIR: Path = Path(
        os.getenv(
            "YOLO_MODEL_DIR",
            "models/artifacts/object_detection/yolov8n/v1.0.0/pytorch",
        )
    )

    # Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/webp"]


settings = Settings()
