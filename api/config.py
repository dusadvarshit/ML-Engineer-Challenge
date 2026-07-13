"""API configuration."""

from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Application settings loaded from the environment."""

    API_TITLE: str = 'ML Engineer Challenge API'
    API_VERSION: str = '1.0.0'
    API_PREFIX: str = '/api/v1'

    DATABASE_URL: str = os.getenv(
        'DATABASE_URL',
        'postgresql+psycopg2://user:pass@localhost:5432/mlchallenge',
    )

    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CELERY_BROKER_URL: str = os.getenv('CELERY_BROKER_URL', REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)
    REQUEST_LOGGING_ENABLED: bool = os.getenv('REQUEST_LOGGING_ENABLED', 'true').lower() in {
        '1',
        'true',
        'yes',
        'on',
    }
    REQUEST_LOG_MAX_BODY_LENGTH: int = int(os.getenv('REQUEST_LOG_MAX_BODY_LENGTH', '2048'))
    REQUEST_ID_HEADER_NAME: str = os.getenv('REQUEST_ID_HEADER_NAME', 'X-Request-Id')
    REQUEST_USER_HEADER_NAME: str = os.getenv('REQUEST_USER_HEADER_NAME', 'X-User-Id')

    MODEL_CACHE_TTL: int = 3600
    MAX_BATCH_SIZE: int = 32
    YOLO_MODEL_DIR: Path = Path(
        os.getenv(
            'YOLO_MODEL_DIR',
            'models/artifacts/object_detection/yolov8n/v1.0.0/pytorch',
        )
    )
    DETR_MODEL_DIR: Path = Path(
        os.getenv(
            'DETR_MODEL_DIR',
            'models/artifacts/object_detection/detr_resnet50/v1.0.0/pytorch',
        )
    )
    RETINANET_MODEL_DIR: Path = Path(
        os.getenv(
            'RETINANET_MODEL_DIR',
            'models/artifacts/object_detection/retinanet_resnet50_fpn/v1.0.0/pytorch',
        )
    )

    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-here')
    ALGORITHM: str = 'HS256'
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    RATE_LIMIT_PER_MINUTE: int = 60

    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    ALLOWED_IMAGE_TYPES: list[str] = ['image/jpeg', 'image/png', 'image/webp']


settings = Settings()
