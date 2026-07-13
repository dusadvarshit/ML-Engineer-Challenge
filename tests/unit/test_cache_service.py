"""Unit tests for optional Redis cache lifecycle management."""

from __future__ import annotations

import asyncio
import hashlib

import pytest
from redis.exceptions import RedisError

from api.config import settings
from api.models.object_detection import ObjectDetection
from api.services.cache_service import RedisCacheService

pytestmark = pytest.mark.unit


class FakeRedisClient:
    """Minimal async Redis stub for cache service tests."""

    def __init__(self, *, ping_error: Exception | None = None) -> None:
        self.closed = False
        self.ping_error = ping_error
        self.values: dict[str, str] = {}
        self.setex_calls: list[tuple[str, int, str]] = []
        self.get_error: Exception | None = None
        self.set_error: Exception | None = None

    async def ping(self) -> None:
        """Raise a configured error or report success."""

        if self.ping_error is not None:
            raise self.ping_error

    async def get(self, key: str) -> str | None:
        """Return a cached payload."""

        if self.get_error is not None:
            raise self.get_error
        return self.values.get(key)

    async def setex(self, key: str, ttl: int, payload: str) -> None:
        """Store a cached payload."""

        if self.set_error is not None:
            raise self.set_error
        self.values[key] = payload
        self.setex_calls.append((key, ttl, payload))

    async def aclose(self) -> None:
        """Track whether the client was closed."""

        self.closed = True


def test_startup_degrades_gracefully_when_redis_is_unavailable(mocker) -> None:
    """Redis startup failure should not leave the service in an active state."""

    client = FakeRedisClient(ping_error=RedisError("redis down"))
    mocker.patch("api.services.cache_service.Redis.from_url", return_value=client)

    service = RedisCacheService()

    asyncio.run(service.startup())

    assert service.client is None
    assert service.is_available is False
    assert client.closed is True


def test_build_detection_key_includes_model_version_and_image_hash() -> None:
    """Detection cache keys should include model version and content hash."""

    service = RedisCacheService()
    image_bytes = b"sample-image"
    digest = hashlib.sha256(image_bytes).hexdigest()

    key = service.build_detection_key(image_bytes, "yolov8n-v1.0.0-best.pt")

    assert key == f"detect:yolov8n-v1.0.0-best.pt:{digest}"


def test_shutdown_returns_cleanly_without_active_client() -> None:
    """Shutdown should no-op when Redis was never connected."""

    service = RedisCacheService()

    asyncio.run(service.shutdown())

    assert service.client is None
    assert service.is_available is False


def test_get_detection_returns_none_when_cache_unavailable() -> None:
    """Reads should no-op when no Redis client is active."""

    service = RedisCacheService()

    cached = asyncio.run(service.get_detection(b"image", "version"))

    assert cached is None


def test_get_detection_returns_none_for_missing_cache_entry() -> None:
    """Missing keys should be treated as cache misses."""

    service = RedisCacheService()
    service._client = FakeRedisClient()
    service._is_available = True

    cached = asyncio.run(service.get_detection(b"image", "version"))

    assert cached is None


def test_get_detection_returns_none_when_redis_get_fails() -> None:
    """Redis read failures should degrade to a cache miss."""

    service = RedisCacheService()
    client = FakeRedisClient()
    client.get_error = RedisError("get failed")
    service._client = client
    service._is_available = True

    cached = asyncio.run(service.get_detection(b"image", "version"))

    assert cached is None


def test_get_detection_returns_none_for_invalid_cached_payload() -> None:
    """Invalid cached JSON should be ignored safely."""

    service = RedisCacheService()
    client = FakeRedisClient()
    service._client = client
    service._is_available = True
    client.values[service.build_detection_key(b"image", "version")] = "not-json"

    cached = asyncio.run(service.get_detection(b"image", "version"))

    assert cached is None


def test_set_detection_returns_false_when_cache_unavailable() -> None:
    """Writes should no-op when no Redis client is active."""

    service = RedisCacheService()
    detections = [
        ObjectDetection(
            x1=1.0,
            y1=2.0,
            x2=3.0,
            y2=4.0,
            confidence=0.9,
            class_id=7,
        )
    ]

    stored = asyncio.run(service.set_detection(b"image", "version", detections))

    assert stored is False


def test_set_detection_returns_false_when_redis_write_fails() -> None:
    """Redis write failures should not bubble out of route code."""

    service = RedisCacheService()
    client = FakeRedisClient()
    client.set_error = RedisError("set failed")
    service._client = client
    service._is_available = True
    detections = [
        ObjectDetection(
            x1=1.0,
            y1=2.0,
            x2=3.0,
            y2=4.0,
            confidence=0.9,
            class_id=7,
        )
    ]

    stored = asyncio.run(service.set_detection(b"image", "version", detections))

    assert stored is False


def test_set_and_get_detection_round_trip() -> None:
    """Stored detections should deserialize back into response models."""

    service = RedisCacheService()
    service._client = FakeRedisClient()
    service._is_available = True
    image_bytes = b"sample-image"
    model_version = "yolov8n-v1.0.0-best.pt"
    detections = [
        ObjectDetection(
            x1=1.0,
            y1=2.0,
            x2=3.0,
            y2=4.0,
            confidence=0.9,
            class_id=7,
        )
    ]

    stored = asyncio.run(service.set_detection(image_bytes, model_version, detections))
    cached = asyncio.run(service.get_detection(image_bytes, model_version))

    assert stored is True
    assert cached is not None
    assert [item.model_dump() for item in cached] == [item.model_dump() for item in detections]
    assert service._client.setex_calls[0][1] == settings.MODEL_CACHE_TTL
