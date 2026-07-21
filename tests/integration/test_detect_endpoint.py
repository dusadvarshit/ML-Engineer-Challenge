"""Integration tests for the mounted detect endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from api.routers import object_detection as router_module

pytestmark = pytest.mark.integration


def test_detect_endpoint_translates_service_errors_to_http_500(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_upload_file,
) -> None:
    """Service inference failures should surface as FastAPI 500 responses."""

    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "get_model_version",
        lambda: "yolov8n-v1.0.0-best.pt",
    )
    monkeypatch.setattr(
        router_module.redis_cache_service,
        "get_detection",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "predict",
        lambda _: (_ for _ in ()).throw(RuntimeError("service failed")),
    )

    response = client.post(
        "/api/v1/detect",
        files={"file": sample_upload_file()},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "inference_failed", "message": "service failed"}
    }


def test_detect_endpoint_rejects_non_image_uploads(
    client,
    sample_text_bytes: bytes,
    sample_upload_file,
) -> None:
    """Route validation should reject multipart payloads that are not images."""

    response = client.post(
        "/api/v1/detect",
        files={
            "file": sample_upload_file(
                "notes.txt", "text/plain", sample_text_bytes
            )
        },
    )

    assert response.status_code == 415
    assert response.json() == {
        "error": {
            "code": "unsupported_media_type",
            "message": "Uploaded file must be a JPEG, PNG, or WebP image.",
        }
    }


def test_detect_endpoint_returns_empty_detection_list_when_model_finds_nothing(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_upload_file,
) -> None:
    """The mounted route should serialize an empty inference result cleanly."""

    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "get_model_version",
        lambda: "yolov8n-v1.0.0-best.pt",
    )
    monkeypatch.setattr(
        router_module.redis_cache_service,
        "get_detection",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        router_module.redis_cache_service,
        "set_detection",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "predict",
        lambda _: [],
    )

    response = client.post(
        "/api/v1/detect",
        files={"file": sample_upload_file()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "yolov8n",
        "model_version": "yolov8n-v1.0.0-best.pt",
        "detections": [],
    }
