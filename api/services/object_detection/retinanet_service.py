"""Service for loading a RetinaNet model and producing detections."""

from __future__ import annotations

from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.Image import Image as PILImage

from api.config import settings
from api.models.object_detection import ObjectDetection


class RetinaNetPredictionService:
    """Load a RetinaNet checkpoint and generate detections for images."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._model_path: Path | None = None
        self._is_warmed_up = False
        self._to_tensor: Any | None = None

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

        model, to_tensor, torch = self._load_model()
        image_tensors = [to_tensor(image) for image in images]
        with self._no_grad(torch):
            results = model(image_tensors)
        return [self._normalize_result(result) for result in results]

    def load(self) -> None:
        model, to_tensor, torch = self._load_model()
        self._warm_up(model, to_tensor, torch)

    def get_model_version(self) -> str:
        model_path = self._model_path or self._resolve_model_path()
        return self._format_model_version(model_path)

    def _decode_image(self, image_bytes: bytes) -> PILImage:
        return Image.open(BytesIO(image_bytes)).convert("RGB")

    def _load_runtime_dependencies(self) -> tuple[Any, Any, Any]:
        import torch
        from torchvision.models.detection import retinanet_resnet50_fpn
        from torchvision.transforms.functional import to_tensor

        return torch, retinanet_resnet50_fpn, to_tensor

    def _load_model(self) -> tuple[Any, Any, Any]:
        if self._model is not None and self._to_tensor is not None:
            torch, _, _ = self._load_runtime_dependencies()
            return self._model, self._to_tensor, torch

        model_path = self._resolve_model_path()

        try:
            torch, model_factory, to_tensor = self._load_runtime_dependencies()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Torchvision is required to run RetinaNet predictions. "
                "Install the torchvision package in the serving environment."
            ) from exc

        model = model_factory(weights=None, weights_backbone=None)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()

        self._model = model
        self._model_path = model_path
        self._to_tensor = to_tensor
        return self._model, self._to_tensor, torch

    def _warm_up(self, model: Any, to_tensor: Any, torch: Any) -> None:
        if self._is_warmed_up:
            return

        warmup_image = Image.new("RGB", (640, 640), color=(0, 0, 0))
        with self._no_grad(torch):
            model([to_tensor(warmup_image)])
        self._is_warmed_up = True

    def _resolve_model_path(self) -> Path:
        model_dir = settings.RETINANET_MODEL_DIR
        candidates = sorted(model_dir.glob("*.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"No RetinaNet checkpoint found in {model_dir}. Add a .pth file there first."
            )
        return candidates[0]

    def _format_model_version(self, model_path: Path) -> str:
        if len(model_path.parents) >= 3:
            model_name = model_path.parents[2].name or "model"
            version = model_path.parents[1].name or "unknown"
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


retinanet_prediction_service = RetinaNetPredictionService()
