"""Integration tests for object detection model selection."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from api.models.object_detection import ObjectDetection
from api.services.object_detection.detr_service import detr_prediction_service
from api.services.object_detection.retinanet_service import retinanet_prediction_service

pytestmark = pytest.mark.integration


def test_detect_endpoint_routes_request_to_detr_model(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_upload_file,
) -> None:
    """The detect endpoint should route requests to the requested DETR backend."""

    monkeypatch.setattr(
        "api.routers.object_detection.redis_cache_service.get_detection",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "api.routers.object_detection.redis_cache_service.set_detection",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        detr_prediction_service,
        "get_model_version",
        lambda: "detr_resnet50-v1.0.0-pytorch",
    )
    monkeypatch.setattr(
        detr_prediction_service,
        "predict",
        lambda _: [
            ObjectDetection(
                x1=1.0,
                y1=2.0,
                x2=3.0,
                y2=4.0,
                confidence=0.91,
                class_id=6,
            )
        ],
    )

    response = client.post(
        "/api/v1/detect",
        data={"model": "detr_resnet50"},
        files={"file": sample_upload_file()},
    )

    assert response.status_code == 200
    assert response.json()["detections"][0]["class_id"] == 6


def test_batch_endpoint_routes_request_to_retinanet_model(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_upload_file,
) -> None:
    """The batch endpoint should route requests to the requested RetinaNet backend."""

    monkeypatch.setattr(
        "api.routers.object_detection.redis_cache_service.get_detection",
        AsyncMock(side_effect=[None, None]),
    )
    monkeypatch.setattr(
        "api.routers.object_detection.redis_cache_service.set_detection",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        retinanet_prediction_service,
        "get_model_version",
        lambda: "retinanet_resnet50_fpn-v1.0.0-model.pth",
    )
    monkeypatch.setattr(
        retinanet_prediction_service,
        "predict_batch_from_bytes",
        lambda _: [
            [
                ObjectDetection(
                    x1=10.0,
                    y1=20.0,
                    x2=30.0,
                    y2=40.0,
                    confidence=0.8,
                    class_id=3,
                )
            ],
            [
                ObjectDetection(
                    x1=50.0,
                    y1=60.0,
                    x2=70.0,
                    y2=80.0,
                    confidence=0.7,
                    class_id=4,
                )
            ],
        ],
    )

    response = client.post(
        "/api/v1/batch",
        data={"task": "detect", "model": "retinanet_resnet50_fpn"},
        files=[
            ("files", sample_upload_file("first.png")),
            ("files", sample_upload_file("second.png")),
        ],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["task"] == "detect"
    assert [item["filename"] for item in payload["results"]] == ["first.png", "second.png"]
    assert [item["detections"][0]["class_id"] for item in payload["results"]] == [3, 4]
