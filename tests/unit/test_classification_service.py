"""Unit tests for the classification adapter boundary."""

from __future__ import annotations

import pytest

from api.config import settings
from api.models.classification import ClassificationPrediction
from api.services.classification import (
    ClassificationPredictionService,
    ClassificationUnavailableError,
)

pytestmark = pytest.mark.unit


def test_service_requires_artifact_and_predictor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """An unconfigured model must fail closed instead of producing fake predictions."""

    service = ClassificationPredictionService()
    monkeypatch.setattr(settings, "CLASSIFICATION_MODEL_DIR", tmp_path)

    with pytest.raises(ClassificationUnavailableError):
        service.predict(b"image", 1)


def test_service_calls_registered_predictor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """The adapter receives raw image bytes and the requested top-k value."""

    (tmp_path / "best_model.pth").write_bytes(b"checkpoint")
    (tmp_path / "classes.json").write_text("[]")
    monkeypatch.setattr(settings, "CLASSIFICATION_MODEL_DIR", tmp_path)
    service = ClassificationPredictionService()
    observed: list[tuple[bytes, int]] = []

    def predictor(image: bytes, top_k: int) -> list[ClassificationPrediction]:
        observed.append((image, top_k))
        return [ClassificationPrediction(class_id=4, confidence=0.5)]

    service.register_predictor(predictor)

    assert service.predict(b"image", 3) == [
        ClassificationPrediction(class_id=4, confidence=0.5)
    ]
    assert observed == [(b"image", 3)]
