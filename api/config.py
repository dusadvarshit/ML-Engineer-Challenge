"""API configuration."""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ApiClient:
    """One API client provisioned through deployment configuration."""

    client_id: str
    api_key: str
    tier: str


class Settings:
    """Application settings loaded from the environment."""

    API_TITLE: str = "ML Engineer Challenge API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"

    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://user:pass@localhost:5432/mlchallenge",
    )

    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", REDIS_URL)
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
    REQUEST_LOGGING_ENABLED: bool = os.getenv(
        "REQUEST_LOGGING_ENABLED", "true"
    ).lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    REQUEST_LOG_MAX_BODY_LENGTH: int = int(
        os.getenv("REQUEST_LOG_MAX_BODY_LENGTH", "2048")
    )
    REQUEST_ID_HEADER_NAME: str = os.getenv("REQUEST_ID_HEADER_NAME", "X-Request-Id")
    REQUEST_USER_HEADER_NAME: str = os.getenv("REQUEST_USER_HEADER_NAME", "X-User-Id")
    API_KEY_HEADER_NAME: str = os.getenv("API_KEY_HEADER_NAME", "X-API-Key")
    API_KEYS: str = os.getenv("API_KEYS", "")
    RATE_LIMITS_PER_MINUTE: str = os.getenv(
        "RATE_LIMITS_PER_MINUTE",
        "standard:60,premium:300",
    )

    MODEL_CACHE_TTL: int = 3600
    MAX_BATCH_SIZE: int = 32
    BATCH_RESULT_TTL_SECONDS: int = int(os.getenv("BATCH_RESULT_TTL_SECONDS", "86400"))
    BATCH_TASK_TIME_LIMIT_SECONDS: int = int(
        os.getenv("BATCH_TASK_TIME_LIMIT_SECONDS", "300")
    )
    BATCH_TASK_MAX_RETRIES: int = int(os.getenv("BATCH_TASK_MAX_RETRIES", "2"))
    YOLO_MODEL_DIR: Path = Path(
        os.getenv(
            "YOLO_MODEL_DIR",
            "models/artifacts/object_detection/yolov8n/v1.0.0/pytorch",
        )
    )
    DETR_MODEL_DIR: Path = Path(
        os.getenv(
            "DETR_MODEL_DIR",
            "models/artifacts/object_detection/detr_resnet50/v1.0.0/pytorch",
        )
    )
    RETINANET_MODEL_DIR: Path = Path(
        os.getenv(
            "RETINANET_MODEL_DIR",
            "models/artifacts/object_detection/retinanet_resnet50_fpn/v1.0.0/pytorch",
        )
    )

    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    RATE_LIMIT_PER_MINUTE: int = 60

    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    MAX_IMAGE_DIMENSION: int = int(os.getenv("MAX_IMAGE_DIMENSION", "8192"))
    MAX_IMAGE_PIXELS: int = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp"]
    DEFAULT_OBJECT_DETECTION_MODEL: str = os.getenv(
        "DEFAULT_OBJECT_DETECTION_MODEL", "yolov8n"
    )
    # Classification serving is deliberately opt-in until a deployment-ready
    # artifact and its compatible runtime adapter have been selected.
    CLASSIFICATION_MODEL_DIR: Path | None = (
        Path(os.environ["CLASSIFICATION_MODEL_DIR"])
        if os.getenv("CLASSIFICATION_MODEL_DIR")
        else None
    )
    CLASSIFICATION_MODEL_VERSION: str = os.getenv(
        "CLASSIFICATION_MODEL_VERSION", "v1.0.0"
    )
    CLASSIFICATION_MODEL_NAME: str = os.getenv(
        "CLASSIFICATION_MODEL_NAME", "efficientnet_b0"
    )

    def get_default_object_detection_model(self) -> str:
        """Return the configured detection backend name."""

        return self.DEFAULT_OBJECT_DETECTION_MODEL

    def get_classification_model_dir(self) -> Path | None:
        """Return the explicitly configured classification artifact directory."""

        return self.CLASSIFICATION_MODEL_DIR

    def get_api_clients(self) -> tuple[ApiClient, ...]:
        """Read configured API clients without exposing credentials in code."""

        if not self.API_KEYS:
            return ()
        try:
            configured_clients = json.loads(self.API_KEYS)
        except json.JSONDecodeError as exc:
            raise ValueError("API_KEYS must be a JSON object.") from exc
        if not isinstance(configured_clients, dict):
            raise ValueError("API_KEYS must be a JSON object.")

        clients: list[ApiClient] = []
        for client_id, value in configured_clients.items():
            if not isinstance(client_id, str) or not isinstance(value, dict):
                raise ValueError(
                    "API_KEYS entries must be client IDs with object values."
                )
            api_key = value.get("api_key")
            tier = value.get("tier", "standard")
            if not isinstance(api_key, str) or not api_key or not isinstance(tier, str):
                raise ValueError(
                    "Each API_KEYS entry requires api_key and optional tier strings."
                )
            clients.append(ApiClient(client_id=client_id, api_key=api_key, tier=tier))
        return tuple(clients)

    def get_rate_limits(self) -> dict[str, int]:
        """Return positive per-minute limits indexed by API-client tier."""

        limits: dict[str, int] = {}
        for item in self.RATE_LIMITS_PER_MINUTE.split(","):
            tier, separator, raw_limit = item.partition(":")
            if not separator or not tier or not raw_limit:
                raise ValueError("RATE_LIMITS_PER_MINUTE must use tier:limit pairs.")
            limit = int(raw_limit)
            if limit <= 0:
                raise ValueError("Rate limits must be positive integers.")
            limits[tier] = limit
        return limits


settings = Settings()
