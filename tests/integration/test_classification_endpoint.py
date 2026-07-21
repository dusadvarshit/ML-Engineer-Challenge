"""Integration tests for the mounted classification endpoint."""

from __future__ import annotations

import pytest

from api.config import settings
from api.models.classification import ClassificationPrediction
from api.routers import classification as router_module
from api.services.classification import classification_prediction_service

pytestmark = pytest.mark.integration


def test_classify_returns_503_when_no_serving_adapter_is_configured(
    client,
    sample_upload_file,
) -> None:
    """A checkpoint directory alone must not pretend to be a serving model."""

    classification_prediction_service.clear_predictor()
    response = client.post(
        "/api/v1/classify", files={"file": sample_upload_file()}
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_classify_returns_adapter_predictions(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_upload_file,
    tmp_path,
) -> None:
    """A configured artifact plus adapter should expose a stable response contract."""

    (tmp_path / "best_model.pth").write_bytes(b"test-checkpoint")
    (tmp_path / "classes.json").write_text('["cat"]')
    monkeypatch.setattr(settings, "CLASSIFICATION_MODEL_DIR", tmp_path)
    monkeypatch.setattr(settings, "CLASSIFICATION_MODEL_VERSION", "test-v1")
    monkeypatch.setattr(
        classification_prediction_service,
        "_predictor",
        lambda _image, top_k: [
            ClassificationPrediction(class_id=0, label="cat", confidence=0.9)
        ][:top_k],
    )
    observations: list[dict[str, object]] = []
    monkeypatch.setattr(
        router_module,
        "observe_inference",
        lambda **kwargs: observations.append(kwargs),
    )

    response = client.post(
        "/api/v1/classify",
        data={"top_k": "1"},
        files={"file": sample_upload_file()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "model": "efficientnet_b0",
        "model_version": "test-v1",
        "predictions": [{"class_id": 0, "label": "cat", "confidence": 0.9}],
    }
    assert observations[0]["task"] == "classify"
    assert observations[0]["outcome"] == "success"


def test_classify_rejects_unconfigured_model(
    client,
    sample_upload_file,
) -> None:
    """Only the configuration-selected model can be requested at runtime."""

    response = client.post(
        "/api/v1/classify",
        data={"model": "resnet50"},
        files={"file": sample_upload_file()},
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "model_unavailable",
            "message": "Classification model 'resnet50' is not configured.",
        }
    }


def test_classify_uses_shared_image_validation(
    client, sample_upload_file
) -> None:
    """Classification should reject unsupported upload content types before inference."""

    response = client.post(
        "/api/v1/classify",
        files={
            "file": sample_upload_file("document.txt", "text/plain", b"text")
        },
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"
