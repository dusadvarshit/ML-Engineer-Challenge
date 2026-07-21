"""Object detection model registry."""

from __future__ import annotations

from typing import Protocol

from api.config import settings
from api.models.object_detection import ObjectDetection, ObjectDetectionModel
from api.models.model_metadata import ModelMetadata
from api.services.classification import classification_prediction_service
from api.services.object_detection.detr_service import detr_prediction_service
from api.services.object_detection.retinanet_service import (
    retinanet_prediction_service,
)
from api.services.object_detection.yolo_service import yolo_prediction_service


class ObjectDetectionPredictionService(Protocol):
    """Minimal contract required by the object detection router."""

    def predict(self, image_bytes: bytes) -> list[ObjectDetection]: ...

    def predict_batch_from_bytes(
        self,
        image_payloads: list[bytes],
    ) -> list[list[ObjectDetection]]: ...

    def load(self) -> None: ...

    def get_model_version(self) -> str: ...


_OBJECT_DETECTION_SERVICES: dict[
    ObjectDetectionModel,
    ObjectDetectionPredictionService,
] = {
    ObjectDetectionModel.YOLOV8N: yolo_prediction_service,
    ObjectDetectionModel.DETR_RESNET50: detr_prediction_service,
    ObjectDetectionModel.RETINANET_RESNET50_FPN: retinanet_prediction_service,
}


def get_object_detection_service(
    model: ObjectDetectionModel,
) -> ObjectDetectionPredictionService:
    """Return the singleton service for the requested object detection model."""

    return _OBJECT_DETECTION_SERVICES[model]


def get_object_detection_model_metadata() -> list[ModelMetadata]:
    """Describe configured model artifacts without exposing local paths."""

    metadata: list[ModelMetadata] = []
    for model, service in _OBJECT_DETECTION_SERVICES.items():
        try:
            version = service.get_model_version()
            artifact_available = True
        except (FileNotFoundError, RuntimeError):
            version = None
            artifact_available = False
        loaded = getattr(service, "_model", None) is not None
        metadata.append(
            ModelMetadata(
                name=model.value,
                version=version,
                artifact_available=artifact_available,
                loaded=loaded,
                ready=artifact_available and loaded,
            )
        )
    return metadata


def get_model_metadata() -> list[ModelMetadata]:
    """Describe all configured inference models without exposing filesystem paths."""

    classification_name = settings.CLASSIFICATION_MODEL_NAME
    classification_metadata = ModelMetadata(
        name=classification_name,
        task="classification",
        version=classification_prediction_service.get_model_version(),
        artifact_available=classification_prediction_service.artifact_available,
        loaded=classification_prediction_service.is_available,
        ready=classification_prediction_service.is_available,
    )
    return [*get_object_detection_model_metadata(), classification_metadata]
