"""Unit tests for optional Redis cache lifecycle management."""

from __future__ import annotations

import asyncio
import hashlib

from redis.exceptions import RedisError

from api.config import settings
from api.models.object_detection import ObjectDetection
from api.services.cache_service import RedisCacheService


class FakeRedisClient:
    """Minimal async Redis stub for cache service tests."""

    def __init__(self, *, ping_error: Exception | None = None) -> None:
        self.closed = False
        self.ping_error = ping_error
        self.values: dict[str, str] = {}
        self.setex_calls: list[tuple[str, int, str]] = []

    async def ping(self) -> None:
        """Raise a configured error or report success."""

        if self.ping_error is not None:
            raise self.ping_error

    async def get(self, key: str) -> str | None:
        """Return a cached payload."""

        return self.values.get(key)

    async def setex(self, key: str, ttl: int, payload: str) -> None:
        """Store a cached payload."""

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
