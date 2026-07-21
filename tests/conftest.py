"""Shared pytest fixtures for API tests."""

from __future__ import annotations

import base64
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.models.object_detection import ObjectDetection

_SAMPLE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC"
)


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch):
    """Return the FastAPI app with startup dependencies disabled."""

    async def fake_cache_startup() -> None:
        return None

    async def fake_cache_shutdown() -> None:
        return None

    monkeypatch.setattr(
        main_module.redis_cache_service, "startup", fake_cache_startup
    )
    monkeypatch.setattr(
        main_module.redis_cache_service, "shutdown", fake_cache_shutdown
    )
    monkeypatch.setattr(
        main_module.yolo_prediction_service, "load", lambda: None
    )
    monkeypatch.setattr(
        main_module.settings,
        "API_KEYS",
        '{"test-client":{"api_key":"test-api-key","tier":"standard"}}',
    )
    return main_module.app


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    """Return a TestClient for the FastAPI app."""

    with TestClient(app) as test_client:
        test_client.headers[main_module.settings.API_KEY_HEADER_NAME] = (
            "test-api-key"
        )
        yield test_client


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return a tiny valid PNG payload for upload tests."""

    return _SAMPLE_PNG


@pytest.fixture
def sample_upload_file(sample_image_bytes: bytes):
    """Build one multipart upload tuple for FastAPI client requests."""

    def build(
        filename: str = "image.png",
        content_type: str = "image/png",
        payload: bytes | None = None,
    ) -> tuple[str, bytes, str]:
        return (
            filename,
            sample_image_bytes if payload is None else payload,
            content_type,
        )

    return build


@pytest.fixture
def sample_text_bytes() -> bytes:
    """Return a non-image payload for validation tests."""

    return b"plain-text-payload"


@pytest.fixture
def sample_detection() -> ObjectDetection:
    """Return one representative object detection."""

    return ObjectDetection(
        x1=1.0,
        y1=2.0,
        x2=10.0,
        y2=20.0,
        confidence=0.95,
        class_id=7,
    )
