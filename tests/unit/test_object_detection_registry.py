"""Unit tests for the object detection service registry."""

from __future__ import annotations

import pytest

from api.models.object_detection import ObjectDetectionModel
from api.services.object_detection.detr_service import detr_prediction_service
from api.services.object_detection.registry import get_object_detection_service
from api.services.object_detection.retinanet_service import (
    retinanet_prediction_service,
)
from api.services.object_detection.yolo_service import yolo_prediction_service

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("model", "expected_service"),
    [
        (ObjectDetectionModel.YOLOV8N, yolo_prediction_service),
        (ObjectDetectionModel.DETR_RESNET50, detr_prediction_service),
        (
            ObjectDetectionModel.RETINANET_RESNET50_FPN,
            retinanet_prediction_service,
        ),
    ],
)
def test_get_object_detection_service_returns_expected_singleton(
    model, expected_service
) -> None:
    """Each object detection model should resolve to its singleton service."""

    assert get_object_detection_service(model) is expected_service
