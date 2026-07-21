"""Adapter boundary for classification inference.

The application ships no selected, runtime-compatible classifier by default.
Keeping loading behind this small adapter makes the API honest about that state
while allowing a deployment to register a tested predictor without changing its
HTTP contract.
"""

from __future__ import annotations

from typing import Protocol

from api.config import settings
from api.models.classification import ClassificationPrediction


class ClassificationUnavailableError(RuntimeError):
    """Raised when classification has no configured, loaded serving artifact."""


class ClassificationPredictor(Protocol):
    """Contract implemented by a deployment-specific model adapter."""

    def __call__(
        self, image_bytes: bytes, top_k: int
    ) -> list[ClassificationPrediction]: ...


class ClassificationPredictionService:
    """Expose one configurable classification model through an injectable adapter."""

    def __init__(self) -> None:
        self._predictor: ClassificationPredictor | None = None

    def register_predictor(self, predictor: ClassificationPredictor) -> None:
        """Register a loaded, validated model adapter (normally during startup)."""

        self._predictor = predictor

    def clear_predictor(self) -> None:
        """Remove the current adapter, primarily for controlled shutdown/tests."""

        self._predictor = None

    def predict(
        self, image_bytes: bytes, top_k: int
    ) -> list[ClassificationPrediction]:
        """Classify one validated image or report that serving is unavailable."""

        if not self.is_available:
            raise ClassificationUnavailableError(
                "Classification is unavailable: configure a selected model artifact "
                "and register its compatible serving adapter."
            )
        assert self._predictor is not None
        return self._predictor(image_bytes, top_k)

    @property
    def artifact_available(self) -> bool:
        """Whether an operator configured a minimally complete artifact directory."""

        artifact_dir = settings.get_classification_model_dir()
        return bool(
            artifact_dir
            and (artifact_dir / "best_model.pth").is_file()
            and (artifact_dir / "classes.json").is_file()
        )

    @property
    def is_available(self) -> bool:
        """Whether a loaded adapter can safely serve requests."""

        return self.artifact_available and self._predictor is not None

    def get_model_version(self) -> str:
        """Return the externally configured version, never an artifact path."""

        return settings.CLASSIFICATION_MODEL_VERSION


classification_prediction_service = ClassificationPredictionService()
