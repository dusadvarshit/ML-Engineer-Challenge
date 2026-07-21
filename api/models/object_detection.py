"""Response models for object detection endpoints."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class InferenceTask(str, Enum):
    """Supported batch inference task types."""

    CLASSIFY = "classify"
    DETECT = "detect"


class ObjectDetectionModel(str, Enum):
    """Supported object detection model backends."""

    YOLOV8N = "yolov8n"
    DETR_RESNET50 = "detr_resnet50"
    RETINANET_RESNET50_FPN = "retinanet_resnet50_fpn"


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

    model: ObjectDetectionModel
    model_version: str
    detections: list[ObjectDetection]


class BatchObjectDetectionItem(BaseModel):
    """Detection results for one image in a batch request."""

    filename: str
    detections: list[ObjectDetection]


class BatchObjectDetectionResponse(BaseModel):
    """Response payload for batch object detection predictions."""

    task: InferenceTask
    results: list[BatchObjectDetectionItem]


class BatchJobStatus(str, Enum):
    """Lifecycle states for a persisted asynchronous batch job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


class BatchJobItem(BaseModel):
    """One item result, including a failure that did not fail the whole batch."""

    filename: str
    content_type: str | None = None
    size_bytes: int
    detections: list[ObjectDetection] | None = None
    error: str | None = None


class BatchJobAcceptedResponse(BaseModel):
    """Response returned after a batch has been durably queued."""

    job_id: str
    status: BatchJobStatus
    status_url: str
    idempotent_replay: bool = False


class BatchJobStatusResponse(BaseModel):
    """Durable batch-job status and partial item results."""

    job_id: str
    task: InferenceTask
    model: ObjectDetectionModel
    model_version: str | None = None
    status: BatchJobStatus
    attempts: int
    error: str | None = None
    items: list[BatchJobItem]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None
