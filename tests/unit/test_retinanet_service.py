"""Unit tests for the RetinaNet prediction service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from api.services.object_detection.retinanet_service import (
    RetinaNetPredictionService,
)

pytestmark = pytest.mark.unit


class FakeTensor:
    """Tiny stand-in for a tensor-like object."""

    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class FakeNoGrad:
    """Minimal no_grad context manager."""

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    """Small torch stub for weight loading and no_grad behavior."""

    def __init__(self) -> None:
        self.load_calls = []

    def load(self, path, map_location="cpu"):
        self.load_calls.append((path, map_location))
        return {"weights": "loaded"}

    def no_grad(self):
        return FakeNoGrad()


class FakeModel:
    """Small RetinaNet model stub."""

    def __init__(self, results=None) -> None:
        self.results = results or []
        self.calls = []
        self.eval_called = False
        self.state_dict = None

    def __call__(self, image_tensors):
        self.calls.append(image_tensors)
        return self.results

    def load_state_dict(self, state_dict):
        self.state_dict = state_dict

    def eval(self):
        self.eval_called = True


def test_predict_returns_first_batch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-image predictions should unwrap the first batch response."""

    service = RetinaNetPredictionService()
    expected = [[{"mock": "detection"}], []]
    monkeypatch.setattr(
        service, "predict_batch_from_bytes", lambda _: expected
    )

    result = service.predict(b"image-bytes")

    assert result == expected[0]


def test_predict_batch_from_bytes_decodes_all_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw image payloads should be decoded before batch inference."""

    service = RetinaNetPredictionService()
    decoded_images = [object(), object()]

    monkeypatch.setattr(
        service,
        "_decode_image",
        lambda image_bytes: (
            decoded_images[0] if image_bytes == b"one" else decoded_images[1]
        ),
    )
    monkeypatch.setattr(
        service, "predict_batch", lambda images: [list(images)]
    )

    result = service.predict_batch_from_bytes([b"one", b"two"])

    assert result == [decoded_images]


def test_predict_batch_returns_empty_list_for_no_images() -> None:
    """Batch prediction should short-circuit empty input."""

    service = RetinaNetPredictionService()

    assert service.predict_batch([]) == []


def test_decode_image_returns_rgb_image(sample_image_bytes: bytes) -> None:
    """Raw image bytes should be converted into an RGB PIL image."""

    service = RetinaNetPredictionService()

    image = service._decode_image(sample_image_bytes)

    assert image.mode == "RGB"
    assert image.size == (1, 1)


def test_resolve_model_path_prefers_sorted_checkpoint_files(
    tmp_path: Path,
) -> None:
    """The service should return the first sorted RetinaNet checkpoint path."""

    service = RetinaNetPredictionService()
    (tmp_path / "zeta.pth").write_bytes(b"checkpoint")
    first_checkpoint = tmp_path / "alpha.pth"
    first_checkpoint.write_bytes(b"checkpoint")

    from api.services.object_detection import (
        retinanet_service as retinanet_service_module,
    )

    original_dir = retinanet_service_module.settings.RETINANET_MODEL_DIR
    retinanet_service_module.settings.RETINANET_MODEL_DIR = tmp_path
    try:
        resolved = service._resolve_model_path()
    finally:
        retinanet_service_module.settings.RETINANET_MODEL_DIR = original_dir

    assert resolved == first_checkpoint


def test_resolve_model_path_raises_when_no_checkpoint_exists(
    tmp_path: Path,
) -> None:
    """The service should raise a clear error when no RetinaNet checkpoint exists."""

    service = RetinaNetPredictionService()

    from api.services.object_detection import (
        retinanet_service as retinanet_service_module,
    )

    original_dir = retinanet_service_module.settings.RETINANET_MODEL_DIR
    retinanet_service_module.settings.RETINANET_MODEL_DIR = tmp_path
    try:
        with pytest.raises(
            FileNotFoundError, match="No RetinaNet checkpoint found"
        ):
            service._resolve_model_path()
    finally:
        retinanet_service_module.settings.RETINANET_MODEL_DIR = original_dir


def test_predict_batch_normalizes_model_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch inference should normalize RetinaNet outputs into API schema objects."""

    service = RetinaNetPredictionService()
    fake_torch = FakeTorch()
    fake_model = FakeModel(
        results=[
            {
                "boxes": FakeTensor([[1, 2, 3, 4]]),
                "scores": FakeTensor([0.9]),
                "labels": FakeTensor([5]),
            }
        ]
    )

    monkeypatch.setattr(
        service,
        "_load_model",
        lambda: (
            fake_model,
            lambda image: f"tensor:{image.width}x{image.height}",
            fake_torch,
        ),
    )

    predictions = service.predict_batch([SimpleNamespace(width=20, height=10)])

    assert len(predictions) == 1
    assert predictions[0][0].model_dump() == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
        "confidence": 0.9,
        "class_id": 5,
    }
    assert fake_model.calls == [["tensor:20x10"]]


def test_load_warms_up_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Service load should trigger both model loading and warmup."""

    service = RetinaNetPredictionService()
    fake_model = FakeModel()
    fake_torch = FakeTorch()
    warmup_calls = []

    monkeypatch.setattr(
        service,
        "_load_model",
        lambda: (fake_model, lambda image: image, fake_torch),
    )
    monkeypatch.setattr(
        service,
        "_warm_up",
        lambda model, to_tensor, torch: warmup_calls.append(
            (model, to_tensor, torch)
        ),
    )

    service.load()

    assert warmup_calls == [(fake_model, warmup_calls[0][1], fake_torch)]


def test_get_model_version_uses_cached_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model version should not resolve the filesystem again once the path is cached."""

    service = RetinaNetPredictionService()
    service._model_path = Path(
        "models/artifacts/object_detection/retinanet_resnet50_fpn/v1.0.0/pytorch/model.pth"
    )

    monkeypatch.setattr(
        service,
        "_resolve_model_path",
        lambda: (_ for _ in ()).throw(AssertionError),
    )

    assert (
        service.get_model_version()
        == "retinanet_resnet50_fpn-v1.0.0-model.pth"
    )


def test_normalize_result_uses_defaults_when_scores_or_labels_missing() -> (
    None
):
    """Missing optional result data should fall back to API-safe defaults."""

    service = RetinaNetPredictionService()

    detections = service._normalize_result(
        {"boxes": FakeTensor([[1, 2, 3, 4]])}
    )

    assert detections[0].confidence == 0.0
    assert detections[0].class_id == -1


def test_load_model_raises_runtime_error_when_torchvision_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing torchvision should produce a serving-friendly runtime error."""

    service = RetinaNetPredictionService()
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(service, "_resolve_model_path", lambda: model_path)

    def raise_missing_dependency():
        raise ModuleNotFoundError("No module named torchvision")

    monkeypatch.setattr(
        service, "_load_runtime_dependencies", raise_missing_dependency
    )

    with pytest.raises(RuntimeError, match="Torchvision is required"):
        service._load_model()


def test_load_model_caches_loaded_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The service should instantiate the RetinaNet model once and reuse it."""

    service = RetinaNetPredictionService()
    model_path = tmp_path / "model.pth"
    model_path.write_bytes(b"checkpoint")
    fake_torch = FakeTorch()
    fake_model = FakeModel()
    factory_calls = []

    def fake_factory(*, weights=None, weights_backbone=None):
        factory_calls.append((weights, weights_backbone))
        return fake_model

    monkeypatch.setattr(service, "_resolve_model_path", lambda: model_path)
    monkeypatch.setattr(
        service,
        "_load_runtime_dependencies",
        lambda: (fake_torch, fake_factory, lambda image: image),
    )

    first_model, first_to_tensor, first_torch = service._load_model()
    second_model, second_to_tensor, second_torch = service._load_model()

    assert first_model is second_model
    assert first_to_tensor is second_to_tensor
    assert first_torch is second_torch
    assert factory_calls == [(None, None)]
    assert fake_torch.load_calls == [(model_path, "cpu")]
    assert fake_model.state_dict == {"weights": "loaded"}
    assert fake_model.eval_called is True
    assert service._model_path == model_path


def test_warm_up_runs_only_once() -> None:
    """Warm-up should not perform duplicate startup predictions."""

    service = RetinaNetPredictionService()
    fake_torch = FakeTorch()
    fake_model = FakeModel()

    service._warm_up(fake_model, lambda image: image, fake_torch)
    service._warm_up(fake_model, lambda image: image, fake_torch)

    assert len(fake_model.calls) == 1
    assert service._is_warmed_up is True
