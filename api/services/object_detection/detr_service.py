"""Service for loading a DETR model and producing detections."""

from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.Image import Image as PILImage

from api.config import settings
from api.models.object_detection import ObjectDetection


class DetrPredictionService:
    """Load a DETR checkpoint and generate detections for images."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._processor: Any | None = None
        self._model_path: Path | None = None
        self._is_warmed_up = False

    def predict(self, image_bytes: bytes) -> list[ObjectDetection]:
        detections = self.predict_batch_from_bytes([image_bytes])
        return detections[0] if detections else []

    def predict_batch_from_bytes(
        self,
        image_payloads: list[bytes],
    ) -> list[list[ObjectDetection]]:
        images = [self._decode_image(image_bytes) for image_bytes in image_payloads]
        return self.predict_batch(images)

    def predict_batch(self, images: list[PILImage]) -> list[list[ObjectDetection]]:
        if not images:
            return []

        model, processor = self._load_model()
        torch = self._load_torch()
        inputs = processor(images=images, return_tensors="pt")
        target_sizes = torch.tensor([[image.height, image.width] for image in images])

        with self._no_grad(torch):
            outputs = model(**inputs)

        results = processor.post_process_object_detection(
            outputs,
            threshold=0.5,
            target_sizes=target_sizes,
        )
        return [self._normalize_result(result) for result in results]

    def load(self) -> None:
        model, processor = self._load_model()
        self._warm_up(model, processor)

    def get_model_version(self) -> str:
        model_path = self._model_path or self._resolve_model_path()
        return self._format_model_version(model_path)

    def _decode_image(self, image_bytes: bytes) -> PILImage:
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _load_torch(self) -> Any:
        import torch

        return torch

    def _load_runtime_dependencies(self) -> tuple[Any, Any]:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        return AutoImageProcessor, AutoModelForObjectDetection

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._processor is not None:
            return self._model, self._processor

        model_path = self._resolve_model_path()

        try:
            image_processor_cls, model_cls = self._load_runtime_dependencies()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Transformers is required to run DETR predictions. "
                "Install the transformers package in the serving environment."
            ) from exc

        self._processor = image_processor_cls.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self._model = model_cls.from_pretrained(
            str(model_path),
            local_files_only=True,
        )
        self._model.eval()
        self._model_path = model_path
        return self._model, self._processor

    def _warm_up(self, model: Any, processor: Any) -> None:
        if self._is_warmed_up:
            return

        torch = self._load_torch()
        warmup_image = Image.new("RGB", (640, 640), color=(0, 0, 0))
        inputs = processor(images=[warmup_image], return_tensors="pt")
        with self._no_grad(torch):
            model(**inputs)
        self._is_warmed_up = True

    def _resolve_model_path(self) -> Path:
        model_dir = settings.DETR_MODEL_DIR
        required_files = [
            model_dir / "config.json",
            model_dir / "model.safetensors",
            model_dir / "preprocessor_config.json",
        ]
        missing_files = [path.name for path in required_files if not path.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing DETR model artifacts in {model_dir}: {', '.join(missing_files)}"
            )
        return model_dir

    def _format_model_version(self, model_path: Path) -> str:
        if len(model_path.parents) >= 2:
            model_name = model_path.parents[1].name or "model"
            version = model_path.parents[0].name or "unknown"
            return f"{model_name}-{version}-{model_path.name}"

        return model_path.name

    def _normalize_result(self, result: dict[str, Any]) -> list[ObjectDetection]:
        boxes = result.get("boxes")
        if boxes is None:
            return []

        xyxy = boxes.cpu().tolist()
        scores = result.get("scores")
        labels = result.get("labels")
        confidences = scores.cpu().tolist() if scores is not None else []
        class_ids = labels.cpu().tolist() if labels is not None else []

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

    def _no_grad(self, torch: Any) -> Any:
        no_grad = getattr(torch, "no_grad", None)
        return no_grad() if callable(no_grad) else nullcontext()


detr_prediction_service = DetrPredictionService()
