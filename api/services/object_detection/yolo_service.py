"""Service for loading a YOLO model and producing detections."""

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.Image import Image as PILImage

from api.config import settings
from api.models.object_detection import ObjectDetection


class YoloPredictionService:
    """Load a YOLO checkpoint and generate detections for images."""

    def __init__(self) -> None:
        """Initialize the service with an empty model cache."""

        self._model: Any | None = None
        self._model_path: Path | None = None
        self._is_warmed_up = False

    def predict(self, image_bytes: bytes) -> list[ObjectDetection]:
        """Run inference on one raw image payload."""

        detections = self.predict_batch_from_bytes([image_bytes])
        return detections[0] if detections else []

    def predict_batch_from_bytes(
        self, image_payloads: list[bytes]
    ) -> list[list[ObjectDetection]]:
        """Decode and run inference on multiple image payloads in one model call."""

        images = [
            self._decode_image(image_bytes) for image_bytes in image_payloads
        ]
        return self.predict_batch(images)

    def predict_batch(
        self, images: list[PILImage]
    ) -> list[list[ObjectDetection]]:
        """Run inference on multiple decoded images in a single YOLO call."""

        if not images:
            return []

        model = self._load_model()
        results = model.predict(images, verbose=False)
        return [self._normalize_result(result) for result in results]

    def load(self) -> None:
        """Preload the configured YOLO model into memory and warm it up."""

        model = self._load_model()
        self._warm_up(model)

    def get_model_version(self) -> str:
        """Return a stable model identifier for cache key generation."""

        model_path = self._model_path or self._resolve_model_path()
        return self._format_model_version(model_path)

    def _decode_image(self, image_bytes: bytes) -> PILImage:
        """Decode raw image bytes into an RGB PIL image."""

        return Image.open(BytesIO(image_bytes)).convert("RGB")

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

    def _warm_up(self, model: Any) -> None:
        """Run one startup inference so the first request avoids warmup cost."""

        if self._is_warmed_up:
            return

        warmup_image = Image.new("RGB", (640, 640), color=(0, 0, 0))
        model.predict(warmup_image, verbose=False)
        self._is_warmed_up = True

    def _resolve_model_path(self) -> Path:
        """Resolve the first available PyTorch checkpoint in the model directory."""

        model_dir = settings.YOLO_MODEL_DIR
        candidates = sorted(model_dir.glob("*.pt")) + sorted(
            model_dir.glob("*.pth")
        )
        if not candidates:
            raise FileNotFoundError(
                f"No PyTorch YOLO checkpoint found in {model_dir}. "
                "Add a .pt or .pth file there first."
            )
        return candidates[0]

    def _format_model_version(self, model_path: Path) -> str:
        """Format a human-readable model version string for cache keys."""

        if len(model_path.parents) >= 3:
            model_name = model_path.parents[2].name or "model"
            version = model_path.parents[1].name or "unknown"
            return f"{model_name}-{version}-{model_path.name}"

        return model_path.name

    def _normalize_result(self, result: Any) -> list[ObjectDetection]:
        """Convert one Ultralytics result object into API response models."""

        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        confidences = (
            boxes.conf.cpu().tolist() if boxes.conf is not None else []
        )
        class_ids = boxes.cls.cpu().tolist() if boxes.cls is not None else []

        detections: list[ObjectDetection] = []
        for index, coords in enumerate(xyxy):
            detections.append(
                ObjectDetection(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                    confidence=(
                        float(confidences[index])
                        if index < len(confidences)
                        else 0.0
                    ),
                    class_id=(
                        int(class_ids[index]) if index < len(class_ids) else -1
                    ),
                )
            )

        return detections


yolo_prediction_service = YoloPredictionService()
