"""Unit tests for the YOLO prediction service."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from api.services.object_detection.yolo_service import YoloPredictionService

pytestmark = pytest.mark.unit


class FakeTensor:
    """Tiny stand-in for a tensor-like object returned by Ultralytics."""

    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class FakeBoxes:
    """Minimal boxes container with tensor-like attributes."""

    def __init__(self, xyxy, conf=None, cls=None):
        self.xyxy = FakeTensor(xyxy)
        self.conf = FakeTensor(conf) if conf is not None else None
        self.cls = FakeTensor(cls) if cls is not None else None


class FakeResult:
    """Minimal Ultralytics result wrapper."""

    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """Test double for the YOLO model instance."""

    def __init__(self, results=None):
        self.results = results or []
        self.predict_calls = []

    def predict(self, images, verbose=False):
        self.predict_calls.append((images, verbose))
        return self.results


def test_predict_returns_first_batch_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-image predictions should unwrap the first batch response."""

    service = YoloPredictionService()
    expected = [[{"mock": "detection"}], []]
    monkeypatch.setattr(
        service, "predict_batch_from_bytes", lambda _: expected
    )

    result = service.predict(b"image-bytes")

    assert result == expected[0]


def test_predict_batch_from_bytes_decodes_all_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw batch payloads should be decoded before batch inference."""

    service = YoloPredictionService()
    decoded_images = [object(), object()]

    monkeypatch.setattr(
        service,
        "_decode_image",
        lambda image_bytes: (
            decoded_images[0] if image_bytes == b"one" else decoded_images[1]
        ),
    )
    monkeypatch.setattr(
        service,
        "predict_batch",
        lambda images: [list(images)],
    )

    result = service.predict_batch_from_bytes([b"one", b"two"])

    assert result == [decoded_images]


def test_predict_batch_returns_empty_list_for_no_images() -> None:
    """Batch prediction should short-circuit empty input."""

    service = YoloPredictionService()

    assert service.predict_batch([]) == []


def test_decode_image_returns_rgb_image(sample_image_bytes: bytes) -> None:
    """Raw image bytes should be converted into an RGB PIL image."""

    service = YoloPredictionService()

    image = service._decode_image(sample_image_bytes)

    assert image.mode == "RGB"
    assert image.size == (1, 1)


def test_resolve_model_path_prefers_sorted_checkpoint_files(
    tmp_path: Path,
) -> None:
    """The service should return the first sorted PyTorch checkpoint path."""

    service = YoloPredictionService()
    (tmp_path / "zeta.pth").write_bytes(b"checkpoint")
    first_checkpoint = tmp_path / "alpha.pt"
    first_checkpoint.write_bytes(b"checkpoint")

    from api.services.object_detection import (
        yolo_service as yolo_service_module,
    )

    original_dir = yolo_service_module.settings.YOLO_MODEL_DIR
    yolo_service_module.settings.YOLO_MODEL_DIR = tmp_path
    try:
        resolved = service._resolve_model_path()
    finally:
        yolo_service_module.settings.YOLO_MODEL_DIR = original_dir

    assert resolved == first_checkpoint


def test_resolve_model_path_raises_when_no_checkpoint_exists(
    tmp_path: Path,
) -> None:
    """The service should raise a clear error when no YOLO checkpoint exists."""

    service = YoloPredictionService()

    from api.services.object_detection import (
        yolo_service as yolo_service_module,
    )

    original_dir = yolo_service_module.settings.YOLO_MODEL_DIR
    yolo_service_module.settings.YOLO_MODEL_DIR = tmp_path
    try:
        with pytest.raises(
            FileNotFoundError, match="No PyTorch YOLO checkpoint found"
        ):
            service._resolve_model_path()
    finally:
        yolo_service_module.settings.YOLO_MODEL_DIR = original_dir


def test_predict_batch_normalizes_model_results() -> None:
    """Batch inference should normalize model outputs into API schema objects."""

    service = YoloPredictionService()
    fake_result = FakeResult(FakeBoxes([[1, 2, 3, 4]], conf=[0.9], cls=[5]))
    fake_model = FakeModel(results=[fake_result])
    service._model = fake_model

    predictions = service.predict_batch([object()])

    assert len(predictions) == 1
    assert predictions[0][0].model_dump() == {
        "x1": 1.0,
        "y1": 2.0,
        "x2": 3.0,
        "y2": 4.0,
        "confidence": 0.9,
        "class_id": 5,
    }
    assert fake_model.predict_calls[0][1] is False


def test_load_warms_up_loaded_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Service load should trigger both model loading and warmup."""

    service = YoloPredictionService()
    fake_model = FakeModel()
    warmup_calls = []

    monkeypatch.setattr(service, "_load_model", lambda: fake_model)
    monkeypatch.setattr(
        service, "_warm_up", lambda model: warmup_calls.append(model)
    )

    service.load()

    assert warmup_calls == [fake_model]


def test_get_model_version_falls_back_to_model_filename() -> None:
    """Shallow model paths should use the checkpoint filename as the version key."""

    service = YoloPredictionService()

    assert service._format_model_version(Path("model.pt")) == "model.pt"


def test_get_model_version_uses_cached_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model version should not resolve the filesystem again once the model path is cached."""

    service = YoloPredictionService()
    service._model_path = Path(
        "models/artifacts/object_detection/yolov8n/v1.0.0/pytorch/model.pt"
    )

    monkeypatch.setattr(
        service,
        "_resolve_model_path",
        lambda: (_ for _ in ()).throw(AssertionError),
    )

    assert service.get_model_version() == "yolov8n-v1.0.0-model.pt"


def test_normalize_result_uses_defaults_when_confidence_or_class_missing() -> (
    None
):
    """Missing optional result data should fall back to API-safe defaults."""

    service = YoloPredictionService()

    detections = service._normalize_result(
        FakeResult(FakeBoxes([[1, 2, 3, 4]]))
    )

    assert detections[0].confidence == 0.0
    assert detections[0].class_id == -1


def test_normalize_result_returns_empty_list_when_boxes_missing() -> None:
    """Results without boxes should serialize as empty detections."""

    service = YoloPredictionService()

    detections = service._normalize_result(types.SimpleNamespace(boxes=None))

    assert detections == []


def test_warm_up_runs_only_once() -> None:
    """Warm-up should not perform duplicate startup predictions."""

    service = YoloPredictionService()
    fake_model = FakeModel()

    service._warm_up(fake_model)
    service._warm_up(fake_model)

    assert len(fake_model.predict_calls) == 1
    assert service._is_warmed_up is True


def test_load_model_raises_runtime_error_when_ultralytics_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing Ultralytics should produce a serving-friendly runtime error."""

    service = YoloPredictionService()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")

    monkeypatch.setattr(service, "_resolve_model_path", lambda: model_path)
    sys.modules.pop("ultralytics", None)

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "ultralytics":
            raise ModuleNotFoundError("No module named ultralytics")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(RuntimeError, match="Ultralytics is required"):
        service._load_model()


def test_load_model_caches_loaded_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The service should instantiate the YOLO model once and reuse it."""

    service = YoloPredictionService()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"checkpoint")
    constructor_calls = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            constructor_calls.append(path)
            self.path = path

    fake_module = types.ModuleType("ultralytics")
    fake_module.YOLO = FakeYOLO

    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)
    monkeypatch.setattr(service, "_resolve_model_path", lambda: model_path)

    first = service._load_model()
    second = service._load_model()

    assert first is second
    assert constructor_calls == [str(model_path)]
    assert service._model_path == model_path
