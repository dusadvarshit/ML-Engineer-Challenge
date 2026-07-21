"""Integration tests for the mounted batch endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from api.models.object_detection import ObjectDetection
from api.routers import object_detection as router_module

pytestmark = pytest.mark.integration


def test_batch_endpoint_preserves_order_and_filenames(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_upload_file,
) -> None:
    """Multipart batch uploads should preserve request ordering in the response."""

    detections = [
        [
            ObjectDetection(
                x1=1.0,
                y1=2.0,
                x2=3.0,
                y2=4.0,
                confidence=0.9,
                class_id=10,
            )
        ],
        [
            ObjectDetection(
                x1=5.0,
                y1=6.0,
                x2=7.0,
                y2=8.0,
                confidence=0.8,
                class_id=20,
            )
        ],
    ]

    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "get_model_version",
        lambda: "yolov8n-v1.0.0-best.pt",
    )
    monkeypatch.setattr(
        router_module.redis_cache_service,
        "get_detection",
        AsyncMock(side_effect=[None, None]),
    )
    monkeypatch.setattr(
        router_module.redis_cache_service,
        "set_detection",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "predict_batch_from_bytes",
        lambda _: detections,
    )

    response = client.post(
        "/api/v1/batch",
        data={"task": "detect"},
        files=[
            ("files", sample_upload_file("first.png")),
            ("files", sample_upload_file("second.png")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {
        "task": "detect",
        "results": [
            {
                "filename": "first.png",
                "detections": [
                    {
                        "x1": 1.0,
                        "y1": 2.0,
                        "x2": 3.0,
                        "y2": 4.0,
                        "confidence": 0.9,
                        "class_id": 10,
                    }
                ],
            },
            {
                "filename": "second.png",
                "detections": [
                    {
                        "x1": 5.0,
                        "y1": 6.0,
                        "x2": 7.0,
                        "y2": 8.0,
                        "confidence": 0.8,
                        "class_id": 20,
                    }
                ],
            },
        ],
    }


def test_batch_endpoint_rejects_unimplemented_classification_task(
    client,
    sample_upload_file,
) -> None:
    """Batch classification should fail explicitly at the HTTP boundary."""

    response = client.post(
        "/api/v1/batch",
        data={"task": "classify"},
        files=[("files", sample_upload_file())],
    )

    assert response.status_code == 501
    assert response.json() == {
        "detail": "Batch classification is not supported yet."
    }


def test_batch_endpoint_rejects_batches_over_route_limit(
    client,
    sample_upload_file,
) -> None:
    """The mounted route should enforce the five-image batch cap."""

    response = client.post(
        "/api/v1/batch",
        data={"task": "detect"},
        files=[
            ("files", sample_upload_file(f"image-{index}.png"))
            for index in range(6)
        ],
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Batch requests support up to 5 images."
    }
