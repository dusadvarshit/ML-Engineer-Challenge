"""Response models for object detection endpoints."""

from pydantic import BaseModel


class ObjectDetection(BaseModel):
    """Single object detection result."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class ObjectDetectionResponse(BaseModel):
    """Response payload for object detection predictions."""

    detections: list[ObjectDetection]
