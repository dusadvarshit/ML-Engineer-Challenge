"""Request and response models for classification endpoints."""

from enum import Enum

from pydantic import BaseModel, Field


class ClassificationModel(str, Enum):
    """Classification architectures supported by the public contract."""

    EFFICIENTNET_B0 = "efficientnet_b0"
    RESNET50 = "resnet50"
    VIT_B_16 = "vit_b_16"


class ClassificationPrediction(BaseModel):
    """Single classification result."""

    class_id: int
    label: str | None = None
    confidence: float = Field(ge=0, le=1)


class ClassificationResponse(BaseModel):
    """Response payload for classification predictions."""

    model: ClassificationModel
    model_version: str
    predictions: list[ClassificationPrediction]
