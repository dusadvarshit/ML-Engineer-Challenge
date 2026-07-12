"""Response models for object detection endpoints."""

from enum import Enum

from pydantic import BaseModel


class InferenceTask(str, Enum):
    """Supported batch inference task types."""

    CLASSIFY = "classify"
    DETECT = "detect"


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


class BatchObjectDetectionItem(BaseModel):
    """Detection results for one image in a batch request."""

    filename: str
    detections: list[ObjectDetection]


class BatchObjectDetectionResponse(BaseModel):
    """Response payload for batch object detection predictions."""

    task: InferenceTask
    results: list[BatchObjectDetectionItem]
