"""Object detection model registry."""

from __future__ import annotations

from typing import Protocol

from api.models.object_detection import ObjectDetection, ObjectDetectionModel
from api.services.object_detection.detr_service import detr_prediction_service
from api.services.object_detection.retinanet_service import retinanet_prediction_service
from api.services.object_detection.yolo_service import yolo_prediction_service


class ObjectDetectionPredictionService(Protocol):
    """Minimal contract required by the object detection router."""

    def predict(self, image_bytes: bytes) -> list[ObjectDetection]:
        ...

    def predict_batch_from_bytes(
        self,
        image_payloads: list[bytes],
    ) -> list[list[ObjectDetection]]:
        ...

    def load(self) -> None:
        ...

    def get_model_version(self) -> str:
        ...


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
