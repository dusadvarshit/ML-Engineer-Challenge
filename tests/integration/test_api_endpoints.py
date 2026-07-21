"""Integration tests for FastAPI endpoint wiring."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.config import settings
from api.models.object_detection import ObjectDetection
from api.routers import object_detection as router_module

pytestmark = pytest.mark.integration


def _patch_lifespan_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load=None,
    startup=None,
    shutdown=None,
) -> None:
    """Patch startup dependencies so app integration tests stay isolated."""

    async def noop_async() -> None:
        return None

    monkeypatch.setattr(
        main_module.redis_cache_service, "startup", startup or noop_async
    )
    monkeypatch.setattr(
        main_module.redis_cache_service, "shutdown", shutdown or noop_async
    )
    monkeypatch.setattr(
        main_module.yolo_prediction_service, "load", load or (lambda: None)
    )


def test_detect_endpoint_uses_app_routing(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_bytes: bytes,
) -> None:
    """The mounted app should expose the object detection endpoint under the API prefix."""

    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setattr(
        settings,
        "API_KEYS",
        '{"test-client":{"api_key":"test-api-key","tier":"standard"}}',
    )
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "predict",
        lambda _: [
            ObjectDetection(
                x1=0.0,
                y1=0.0,
                x2=9.0,
                y2=9.0,
                confidence=0.77,
                class_id=3,
            )
        ],
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/api/v1/detect",
            files={"file": ("image.png", sample_image_bytes, "image/png")},
            headers={settings.API_KEY_HEADER_NAME: "test-api-key"},
        )

    assert response.status_code == 200
    assert response.json()["detections"][0]["class_id"] == 3


def test_startup_failure_from_model_load_bubbles_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App startup errors should fail fast during integration testing."""

    def raise_startup_error() -> None:
        raise RuntimeError("startup failed")

    _patch_lifespan_dependencies(monkeypatch, load=raise_startup_error)

    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(main_module.app):
            pass
