"""Redis lifecycle management for optional cache access."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence

from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.config import settings
from api.models.object_detection import ObjectDetection

logger = logging.getLogger(__name__)


class RedisCacheService:
    """Manage an optional Redis client without making API startup depend on it."""

    def __init__(self) -> None:
        """Initialize the service with no active Redis client."""

        self._client: Redis | None = None
        self._is_available = False

    @property
    def client(self) -> Redis | None:
        """Return the Redis client when it is available."""

        if not self._is_available:
            return None
        return self._client

    @property
    def is_available(self) -> bool:
        """Return whether Redis is currently reachable."""

        return self._is_available

    async def startup(self) -> None:
        """Attempt to connect to Redis and degrade gracefully on failure."""

        client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

        try:
            await client.ping()
        except (OSError, RedisError) as exc:
            logger.warning(
                "Redis unavailable; continuing without cache support: %s",
                exc,
            )
            await self._close_client(client)
            self._client = None
            self._is_available = False
            return

        self._client = client
        self._is_available = True
        logger.info("Redis cache service connected.")

    async def shutdown(self) -> None:
        """Close any active Redis client."""

        self._is_available = False
        if self._client is None:
            return

        await self._close_client(self._client)
        self._client = None

    def build_detection_key(self, image_bytes: bytes, model_version: str) -> str:
        """Build a stable cache key for one detection request."""

        digest = hashlib.sha256(image_bytes).hexdigest()
        return f"detect:{model_version}:{digest}"

    async def get_detection(
        self,
        image_bytes: bytes,
        model_version: str,
    ) -> list[ObjectDetection] | None:
        """Return cached detections for one image when available."""

        client = self.client
        if client is None:
            return None

        key = self.build_detection_key(image_bytes, model_version)

        try:
            payload = await client.get(key)
        except (OSError, RedisError) as exc:
            logger.warning("Redis get failed for %s: %s", key, exc)
            return None

        if payload is None:
            return None

        try:
            cached_items = json.loads(payload)
            return [ObjectDetection.model_validate(item) for item in cached_items]
        except (TypeError, ValueError) as exc:
            logger.warning("Invalid cached detection payload for %s: %s", key, exc)
            return None

    async def set_detection(
        self,
        image_bytes: bytes,
        model_version: str,
        detections: Sequence[ObjectDetection],
    ) -> bool:
        """Store detections in Redis and return whether the write succeeded."""

        client = self.client
        if client is None:
            return False

        key = self.build_detection_key(image_bytes, model_version)
        payload = json.dumps(
            [detection.model_dump() for detection in detections],
            separators=(",", ":"),
        )

        try:
            await client.setex(key, settings.MODEL_CACHE_TTL, payload)
        except (OSError, RedisError) as exc:
            logger.warning("Redis set failed for %s: %s", key, exc)
            return False

        return True

    async def _close_client(self, client: Redis) -> None:
        """Close a Redis client across redis-py versions."""

        close = getattr(client, "aclose", None)
        if close is not None:
            await close()
            return

        await client.close()


redis_cache_service = RedisCacheService()
