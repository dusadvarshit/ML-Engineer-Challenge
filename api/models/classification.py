"""Request and response models for classification endpoints."""

from pydantic import BaseModel


class ClassificationPrediction(BaseModel):
    """Single classification result."""

    class_id: int
    confidence: float


class ClassificationResponse(BaseModel):
    """Response payload for classification predictions."""

    predictions: list[ClassificationPrediction]
