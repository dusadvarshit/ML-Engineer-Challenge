"""Unit tests for object detection route caching behavior."""

from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock

from starlette.datastructures import Headers, UploadFile

from api.models.object_detection import ObjectDetection
from api.routers.object_detection import detect_objects
from api.services.cache_service import redis_cache_service
from api.services.object_detection.yolo_service import yolo_prediction_service



def _build_upload_file(image_bytes: bytes) -> UploadFile:
    """Create an image upload payload for route tests."""

    return UploadFile(
        file=BytesIO(image_bytes),
        filename="test.png",
        headers=Headers({"content-type": "image/png"}),
    )



def test_detect_objects_returns_cached_result_without_running_inference(mocker) -> None:
    """Cache hits should short-circuit the YOLO prediction call."""

    image_bytes = b"cached-image"
    cached_detections = [
        ObjectDetection(
            x1=1.0,
            y1=2.0,
            x2=3.0,
            y2=4.0,
            confidence=0.9,
            class_id=5,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        "get_model_version",
        return_value="yolov8n-v1.0.0-best.pt",
    )
    get_detection = mocker.patch.object(
        redis_cache_service,
        "get_detection",
        new=AsyncMock(return_value=cached_detections),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        "set_detection",
        new=AsyncMock(return_value=False),
    )
    predict = mocker.patch.object(yolo_prediction_service, "predict")

    response = asyncio.run(detect_objects(_build_upload_file(image_bytes)))

    assert [item.model_dump() for item in response.detections] == [
        item.model_dump() for item in cached_detections
    ]
    get_detection.assert_awaited_once_with(image_bytes, "yolov8n-v1.0.0-best.pt")
    set_detection.assert_not_awaited()
    predict.assert_not_called()



def test_detect_objects_caches_fresh_inference_result(mocker) -> None:
    """Cache misses should run inference and store the new detections."""

    image_bytes = b"fresh-image"
    fresh_detections = [
        ObjectDetection(
            x1=10.0,
            y1=20.0,
            x2=30.0,
            y2=40.0,
            confidence=0.8,
            class_id=2,
        )
    ]

    mocker.patch.object(
        yolo_prediction_service,
        "get_model_version",
        return_value="yolov8n-v1.0.0-best.pt",
    )
    get_detection = mocker.patch.object(
        redis_cache_service,
        "get_detection",
        new=AsyncMock(return_value=None),
    )
    set_detection = mocker.patch.object(
        redis_cache_service,
        "set_detection",
        new=AsyncMock(return_value=True),
    )
    predict = mocker.patch.object(
        yolo_prediction_service,
        "predict",
        return_value=fresh_detections,
    )

    response = asyncio.run(detect_objects(_build_upload_file(image_bytes)))

    assert [item.model_dump() for item in response.detections] == [
        item.model_dump() for item in fresh_detections
    ]
    get_detection.assert_awaited_once_with(image_bytes, "yolov8n-v1.0.0-best.pt")
    predict.assert_called_once_with(image_bytes)
    set_detection.assert_awaited_once_with(
        image_bytes,
        "yolov8n-v1.0.0-best.pt",
        fresh_detections,
    )
