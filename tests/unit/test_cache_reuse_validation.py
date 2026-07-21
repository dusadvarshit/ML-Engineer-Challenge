"""Validation tests for repeated-request cache reuse and latency improvement."""

from __future__ import annotations

import time
from io import BytesIO

import pytest
from PIL import Image

from api.models.object_detection import ObjectDetection
from api.routers import object_detection as router_module

pytestmark = pytest.mark.unit


class FakeRedisClient:
    """Minimal async Redis stub for repeated-request cache validation."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        """Return a cached payload for one key."""

        return self.values.get(key)

    async def setex(self, key: str, ttl: int, payload: str) -> None:
        """Store a cached payload for one key."""

        self.values[key] = payload


def _make_png_bytes(color: tuple[int, int, int]) -> bytes:
    """Create a small valid PNG payload with a unique pixel color."""

    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_repeated_detect_request_reuses_cache_and_gets_faster(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_detection: ObjectDetection,
    sample_image_bytes: bytes,
) -> None:
    """The second identical detect request should bypass YOLO and return faster."""

    fake_client = FakeRedisClient()
    monkeypatch.setattr(
        router_module.redis_cache_service, "_client", fake_client
    )
    monkeypatch.setattr(
        router_module.redis_cache_service, "_is_available", True
    )
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "get_model_version",
        lambda: "test-model",
    )

    predict_calls = {"count": 0}

    def slow_predict(_: bytes) -> list[ObjectDetection]:
        predict_calls["count"] += 1
        time.sleep(0.05)
        return [sample_detection]

    monkeypatch.setattr(
        router_module.yolo_prediction_service, "predict", slow_predict
    )

    started_at = time.perf_counter()
    first_response = client.post(
        "/api/v1/detect",
        files={"file": ("image.png", sample_image_bytes, "image/png")},
    )
    first_duration = time.perf_counter() - started_at

    started_at = time.perf_counter()
    second_response = client.post(
        "/api/v1/detect",
        files={"file": ("image.png", sample_image_bytes, "image/png")},
    )
    second_duration = time.perf_counter() - started_at

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == second_response.json()
    assert predict_calls["count"] == 1
    assert second_duration < first_duration * 0.5


def test_repeated_mixed_batch_request_reuses_cache_and_gets_faster(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second identical mixed batch request should avoid batch inference and speed up."""

    image_a = _make_png_bytes((255, 0, 0))
    image_b = _make_png_bytes((0, 255, 0))
    detection_a = ObjectDetection(
        x1=1.0,
        y1=2.0,
        x2=3.0,
        y2=4.0,
        confidence=0.95,
        class_id=1,
    )
    detection_b = ObjectDetection(
        x1=5.0,
        y1=6.0,
        x2=7.0,
        y2=8.0,
        confidence=0.85,
        class_id=2,
    )

    fake_client = FakeRedisClient()
    monkeypatch.setattr(
        router_module.redis_cache_service, "_client", fake_client
    )
    monkeypatch.setattr(
        router_module.redis_cache_service, "_is_available", True
    )
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "get_model_version",
        lambda: "test-model",
    )

    predict_batch_calls = {"count": 0}

    def slow_predict_batch(
        payloads: list[bytes],
    ) -> list[list[ObjectDetection]]:
        predict_batch_calls["count"] += 1
        time.sleep(0.05)
        results: list[list[ObjectDetection]] = []
        for payload in payloads:
            if payload == image_a:
                results.append([detection_a])
            elif payload == image_b:
                results.append([detection_b])
            else:
                raise AssertionError(
                    "Unexpected payload received by batch predictor."
                )
        return results

    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        "predict_batch_from_bytes",
        slow_predict_batch,
    )

    files = [
        ("files", ("first.png", image_a, "image/png")),
        ("files", ("second.png", image_b, "image/png")),
    ]

    started_at = time.perf_counter()
    first_response = client.post(
        "/api/v1/batch", data={"task": "detect"}, files=files
    )
    first_duration = time.perf_counter() - started_at

    started_at = time.perf_counter()
    second_response = client.post(
        "/api/v1/batch", data={"task": "detect"}, files=files
    )
    second_duration = time.perf_counter() - started_at

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == second_response.json()
    assert predict_batch_calls["count"] == 1
    assert second_duration < first_duration * 0.5
