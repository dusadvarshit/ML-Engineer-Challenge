"""Service for loading a YOLO model and producing detections."""

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from api.config import settings
from api.models.object_detection import ObjectDetection


class YoloPredictionService:
    """Load a YOLO checkpoint and generate detections for images."""

    def __init__(self) -> None:
        """Initialize the service with an empty model cache."""

        self._model: Any | None = None
        self._model_path: Path | None = None

    def predict(self, image_bytes: bytes) -> list[ObjectDetection]:
        """Run inference on raw image bytes and return normalized detections."""

        model = self._load_model()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        results = model.predict(image, verbose=False)
        return self._normalize_output(results)

    def _load_model(self) -> Any:
        """Load and cache the configured YOLO checkpoint."""

        if self._model is not None:
            return self._model

        model_path = self._resolve_model_path()

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Ultralytics is required to run YOLO predictions. "
                "Install the ultralytics package in the serving environment."
            ) from exc

        self._model = YOLO(str(model_path))
        self._model_path = model_path
        return self._model

    def _resolve_model_path(self) -> Path:
        """Resolve the first available PyTorch checkpoint in the model directory."""

        model_dir = settings.YOLO_MODEL_DIR
        candidates = sorted(model_dir.glob("*.pt")) + sorted(model_dir.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No PyTorch YOLO checkpoint found in {model_dir}. "
                "Add a .pt or .pth file there first."
            )
        return candidates[0]

    def _normalize_output(self, raw_output: Any) -> list[ObjectDetection]:
        """Convert Ultralytics prediction results into API response models."""

        if not raw_output:
            return []

        result = raw_output[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist() if boxes.conf is not None else []
        class_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else []

        detections: list[ObjectDetection] = []
        for index, coords in enumerate(xyxy):
            detections.append(
                ObjectDetection(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                    confidence=float(confidences[index]) if index < len(confidences) else 0.0,
                    class_id=int(class_ids[index]) if index < len(class_ids) else -1,
                )
            )

        return detections


yolo_prediction_service = YoloPredictionService()
