"""Unit tests for Prometheus inference and cache metrics."""

from __future__ import annotations

import json

import pytest
from prometheus_client import REGISTRY

from api.routers import object_detection as router_module

pytestmark = pytest.mark.unit


class FakeRedisClient:
    """Minimal async Redis stub for cache metrics tests."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        """Return the cached payload for a key."""

        return self.values.get(key)

    async def setex(self, key: str, ttl: int, payload: str) -> None:
        """Store the cached payload for a key."""

        self.values[key] = payload


def test_detect_endpoint_records_inference_metrics(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_detection,
    sample_image_bytes: bytes,
) -> None:
    """Successful detection requests should increment inference counters."""

    labels = {
        'task': 'detect',
        'model': 'test-model',
        'outcome': 'success',
    }
    before_total = REGISTRY.get_sample_value('ml_model_inference_requests_total', labels) or 0.0
    before_duration_count = (
        REGISTRY.get_sample_value('ml_model_inference_duration_seconds_count', labels)
        or 0.0
    )

    monkeypatch.setattr(router_module.yolo_prediction_service, 'get_model_version', lambda: 'test-model')
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        'predict',
        lambda _: [sample_detection],
    )

    response = client.post(
        '/api/v1/detect',
        files={'file': ('image.png', sample_image_bytes, 'image/png')},
    )

    after_total = REGISTRY.get_sample_value('ml_model_inference_requests_total', labels) or 0.0
    after_duration_count = (
        REGISTRY.get_sample_value('ml_model_inference_duration_seconds_count', labels)
        or 0.0
    )

    assert response.status_code == 200
    assert after_total == before_total + 1
    assert after_duration_count == before_duration_count + 1


def test_batch_endpoint_records_batch_metrics(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_detection,
    sample_image_bytes: bytes,
) -> None:
    """Successful batch requests should publish batch counters and sizes."""

    batch_labels = {'task': 'detect', 'outcome': 'success'}
    inference_labels = {
        'task': 'detect',
        'model': 'test-model',
        'outcome': 'success',
    }
    batch_size_labels = {'task': 'detect'}

    before_batch_total = REGISTRY.get_sample_value('ml_batch_requests_total', batch_labels) or 0.0
    before_batch_size_count = REGISTRY.get_sample_value('ml_batch_size_count', batch_size_labels) or 0.0
    before_batch_size_sum = REGISTRY.get_sample_value('ml_batch_size_sum', batch_size_labels) or 0.0
    before_inference_total = (
        REGISTRY.get_sample_value('ml_model_inference_requests_total', inference_labels)
        or 0.0
    )

    monkeypatch.setattr(router_module.yolo_prediction_service, 'get_model_version', lambda: 'test-model')
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        'predict_batch_from_bytes',
        lambda payloads: [[sample_detection] for _ in payloads],
    )

    response = client.post(
        '/api/v1/batch',
        data={'task': 'detect'},
        files=[
            ('files', ('first.png', sample_image_bytes, 'image/png')),
            ('files', ('second.png', sample_image_bytes, 'image/png')),
        ],
    )

    after_batch_total = REGISTRY.get_sample_value('ml_batch_requests_total', batch_labels) or 0.0
    after_batch_size_count = REGISTRY.get_sample_value('ml_batch_size_count', batch_size_labels) or 0.0
    after_batch_size_sum = REGISTRY.get_sample_value('ml_batch_size_sum', batch_size_labels) or 0.0
    after_inference_total = (
        REGISTRY.get_sample_value('ml_model_inference_requests_total', inference_labels)
        or 0.0
    )

    assert response.status_code == 200
    assert after_batch_total == before_batch_total + 1
    assert after_batch_size_count == before_batch_size_count + 1
    assert after_batch_size_sum == before_batch_size_sum + 2
    assert after_inference_total == before_inference_total + 2


def test_detect_endpoint_records_cache_miss_and_store_metrics(
    client,
    monkeypatch: pytest.MonkeyPatch,
    sample_detection,
    sample_image_bytes: bytes,
) -> None:
    """Cache metrics should reflect a lookup miss followed by a successful store."""

    lookup_labels = {'cache': 'detection', 'operation': 'lookup', 'outcome': 'miss'}
    store_labels = {'cache': 'detection', 'operation': 'store', 'outcome': 'success'}
    before_lookup = REGISTRY.get_sample_value('ml_cache_operations_total', lookup_labels) or 0.0
    before_store = REGISTRY.get_sample_value('ml_cache_operations_total', store_labels) or 0.0

    fake_client = FakeRedisClient()
    monkeypatch.setattr(router_module.redis_cache_service, '_client', fake_client)
    monkeypatch.setattr(router_module.redis_cache_service, '_is_available', True)
    monkeypatch.setattr(router_module.yolo_prediction_service, 'get_model_version', lambda: 'test-model')
    monkeypatch.setattr(
        router_module.yolo_prediction_service,
        'predict',
        lambda _: [sample_detection],
    )

    response = client.post(
        '/api/v1/detect',
        files={'file': ('image.png', sample_image_bytes, 'image/png')},
    )

    after_lookup = REGISTRY.get_sample_value('ml_cache_operations_total', lookup_labels) or 0.0
    after_store = REGISTRY.get_sample_value('ml_cache_operations_total', store_labels) or 0.0

    assert response.status_code == 200
    assert after_lookup == before_lookup + 1
    assert after_store == before_store + 1


def test_detect_endpoint_records_cache_hit_metrics_without_inference(
    client,
    monkeypatch: pytest.MonkeyPatch,
    mocker,
    sample_detection,
    sample_image_bytes: bytes,
) -> None:
    """Cache hits should increment hit metrics and skip YOLO inference."""

    lookup_labels = {'cache': 'detection', 'operation': 'lookup', 'outcome': 'hit'}
    before_lookup = REGISTRY.get_sample_value('ml_cache_operations_total', lookup_labels) or 0.0

    fake_client = FakeRedisClient()
    monkeypatch.setattr(router_module.redis_cache_service, '_client', fake_client)
    monkeypatch.setattr(router_module.redis_cache_service, '_is_available', True)
    monkeypatch.setattr(router_module.yolo_prediction_service, 'get_model_version', lambda: 'test-model')
    key = router_module.redis_cache_service.build_detection_key(sample_image_bytes, 'test-model')
    fake_client.values[key] = json.dumps([sample_detection.model_dump()])
    predict = mocker.patch.object(router_module.yolo_prediction_service, 'predict')

    response = client.post(
        '/api/v1/detect',
        files={'file': ('image.png', sample_image_bytes, 'image/png')},
    )

    after_lookup = REGISTRY.get_sample_value('ml_cache_operations_total', lookup_labels) or 0.0

    assert response.status_code == 200
    assert after_lookup == before_lookup + 1
    predict.assert_not_called()
