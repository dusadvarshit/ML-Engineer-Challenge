"""Unit tests for API schema models."""

from __future__ import annotations

import pytest

from api.models.classification import ClassificationPrediction, ClassificationResponse
from api.models.object_detection import (
    BatchObjectDetectionItem,
    BatchObjectDetectionResponse,
    InferenceTask,
    ObjectDetection,
    ObjectDetectionResponse,
)

pytestmark = pytest.mark.unit


def test_object_detection_response_serializes_enum_and_payload() -> None:
    """Object detection responses should serialize the expected nested shape."""

    detection = ObjectDetection(
        x1=1.0,
        y1=2.0,
        x2=3.0,
        y2=4.0,
        confidence=0.5,
        class_id=9,
    )
    response = BatchObjectDetectionResponse(
        task=InferenceTask.DETECT,
        results=[BatchObjectDetectionItem(filename="image.png", detections=[detection])],
    )

    assert response.model_dump() == {
        "task": "detect",
        "results": [
            {
                "filename": "image.png",
                "detections": [
                    {
                        "x1": 1.0,
                        "y1": 2.0,
                        "x2": 3.0,
                        "y2": 4.0,
                        "confidence": 0.5,
                        "class_id": 9,
                    }
                ],
            }
        ],
    }


def test_object_detection_response_wraps_detections() -> None:
    """The single-image response model should preserve detection payloads."""

    detection = ObjectDetection(
        x1=5.0,
        y1=6.0,
        x2=7.0,
        y2=8.0,
        confidence=0.8,
        class_id=1,
    )

    response = ObjectDetectionResponse(detections=[detection])

    assert response.model_dump()["detections"][0]["class_id"] == 1


def test_classification_response_serializes_predictions() -> None:
    """Classification models should validate the placeholder schema correctly."""

    response = ClassificationResponse(
        predictions=[ClassificationPrediction(class_id=42, confidence=0.99)]
    )

    assert response.model_dump() == {
        "predictions": [{"class_id": 42, "confidence": 0.99}]
    }
