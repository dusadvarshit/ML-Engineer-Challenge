"""Unit tests for the DETR prediction service."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from api.services.object_detection.detr_service import DetrPredictionService

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
    """Small torch stub for target-size and no_grad behavior."""

    def __init__(self) -> None:
        self.tensor_calls = []

    def tensor(self, value):
        self.tensor_calls.append(value)
        return value

    def no_grad(self):
        return FakeNoGrad()


class FakeProcessor:
    """Small image processor stub."""

    def __init__(self, results=None) -> None:
        self.results = results or []
        self.calls = []
        self.post_process_calls = []

    def __call__(self, *, images, return_tensors):
        self.calls.append((images, return_tensors))
        return {"pixel_values": images}

    def post_process_object_detection(self, outputs, threshold, target_sizes):
        self.post_process_calls.append((outputs, threshold, target_sizes))
        return self.results


class FakeModel:
    """Small DETR model stub."""

    def __init__(self) -> None:
        self.calls = []
        self.eval_called = False

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return "model-output"

    def eval(self):
        self.eval_called = True


def test_predict_returns_first_batch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-image predictions should unwrap the first batch response."""

    service = DetrPredictionService()
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

    service = DetrPredictionService()
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

    service = DetrPredictionService()

    assert service.predict_batch([]) == []


def test_decode_image_returns_rgb_image(sample_image_bytes: bytes) -> None:
    """Raw image bytes should be converted into an RGB PIL image."""

    service = DetrPredictionService()

    image = service._decode_image(sample_image_bytes)

    assert image.mode == "RGB"
    assert image.size == (1, 1)


def test_resolve_model_path_requires_expected_artifacts(
    tmp_path: Path,
) -> None:
    """The service should resolve the DETR directory only when required files exist."""

    service = DetrPredictionService()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "preprocessor_config.json").write_text("{}")

    from api.services.object_detection import (
        detr_service as detr_service_module,
    )

    original_dir = detr_service_module.settings.DETR_MODEL_DIR
    detr_service_module.settings.DETR_MODEL_DIR = tmp_path
    try:
        resolved = service._resolve_model_path()
    finally:
        detr_service_module.settings.DETR_MODEL_DIR = original_dir

    assert resolved == tmp_path


def test_resolve_model_path_raises_when_artifacts_are_missing(
    tmp_path: Path,
) -> None:
    """The service should raise a clear error when DETR files are incomplete."""

    service = DetrPredictionService()

    from api.services.object_detection import (
        detr_service as detr_service_module,
    )

    original_dir = detr_service_module.settings.DETR_MODEL_DIR
    detr_service_module.settings.DETR_MODEL_DIR = tmp_path
    try:
        with pytest.raises(
            FileNotFoundError, match="Missing DETR model artifacts"
        ):
            service._resolve_model_path()
    finally:
        detr_service_module.settings.DETR_MODEL_DIR = original_dir


def test_predict_batch_normalizes_model_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch inference should normalize DETR outputs into API schema objects."""

    service = DetrPredictionService()
    fake_torch = FakeTorch()
    fake_processor = FakeProcessor(
        results=[
            {
                "boxes": FakeTensor([[1, 2, 3, 4]]),
                "scores": FakeTensor([0.9]),
                "labels": FakeTensor([5]),
            }
        ]
    )
    fake_model = FakeModel()

    monkeypatch.setattr(
        service, "_load_model", lambda: (fake_model, fake_processor)
    )
    monkeypatch.setattr(service, "_load_torch", lambda: fake_torch)

    predictions = service.predict_batch([SimpleNamespace(height=10, width=20)])

    assert len(predictions) == 1
    assert predictions[0][0].model_dump() == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
        "confidence": 0.9,
        "class_id": 5,
    }
    assert fake_torch.tensor_calls == [[[10, 20]]]
    assert fake_processor.post_process_calls[0][1] == 0.5


def test_load_warms_up_loaded_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Service load should trigger both model loading and warmup."""

    service = DetrPredictionService()
    fake_model = FakeModel()
    fake_processor = FakeProcessor()
    warmup_calls = []

    monkeypatch.setattr(
        service, "_load_model", lambda: (fake_model, fake_processor)
    )
    monkeypatch.setattr(
        service,
        "_warm_up",
        lambda model, processor: warmup_calls.append((model, processor)),
    )

    service.load()

    assert warmup_calls == [(fake_model, fake_processor)]


def test_get_model_version_uses_cached_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model version should not hit the filesystem again once the path is cached."""

    service = DetrPredictionService()
    service._model_path = Path(
        "models/artifacts/object_detection/detr_resnet50/v1.0.0/pytorch"
    )

    monkeypatch.setattr(
        service,
        "_resolve_model_path",
        lambda: (_ for _ in ()).throw(AssertionError),
    )

    assert service.get_model_version() == "detr_resnet50-v1.0.0-pytorch"


def test_normalize_result_uses_defaults_when_scores_or_labels_missing() -> (
    None
):
    """Missing optional result data should fall back to API-safe defaults."""

    service = DetrPredictionService()

    detections = service._normalize_result(
        {"boxes": FakeTensor([[1, 2, 3, 4]])}
    )

    assert detections[0].confidence == 0.0
    assert detections[0].class_id == -1


def test_load_model_raises_runtime_error_when_transformers_are_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing transformers should produce a serving-friendly runtime error."""

    service = DetrPredictionService()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "preprocessor_config.json").write_text("{}")

    monkeypatch.setattr(service, "_resolve_model_path", lambda: tmp_path)

    def raise_missing_dependency():
        raise ModuleNotFoundError("No module named transformers")

    monkeypatch.setattr(
        service, "_load_runtime_dependencies", raise_missing_dependency
    )

    with pytest.raises(RuntimeError, match="Transformers is required"):
        service._load_model()


def test_load_model_caches_loaded_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The service should instantiate the DETR artifacts once and reuse them."""

    service = DetrPredictionService()
    load_calls = []

    class FakeImageProcessorClass:
        @classmethod
        def from_pretrained(cls, path: str, local_files_only: bool = False):
            load_calls.append(("processor", path, local_files_only))
            return FakeProcessor()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, path: str, local_files_only: bool = False):
            load_calls.append(("model", path, local_files_only))
            return FakeModel()

    monkeypatch.setattr(service, "_resolve_model_path", lambda: tmp_path)
    monkeypatch.setattr(
        service,
        "_load_runtime_dependencies",
        lambda: (FakeImageProcessorClass, FakeModelClass),
    )

    first_model, first_processor = service._load_model()
    second_model, second_processor = service._load_model()

    assert first_model is second_model
    assert first_processor is second_processor
    assert load_calls == [
        ("processor", str(tmp_path), True),
        ("model", str(tmp_path), True),
    ]
    assert service._model_path == tmp_path
    assert first_model.eval_called is True


def test_warm_up_runs_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Warm-up should not perform duplicate startup inference."""

    service = DetrPredictionService()
    fake_torch = FakeTorch()
    fake_model = FakeModel()
    fake_processor = FakeProcessor()

    monkeypatch.setattr(service, "_load_torch", lambda: fake_torch)

    service._warm_up(fake_model, fake_processor)
    service._warm_up(fake_model, fake_processor)

    assert len(fake_model.calls) == 1
    assert service._is_warmed_up is True
