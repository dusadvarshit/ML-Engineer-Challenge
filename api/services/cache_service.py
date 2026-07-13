"""Redis lifecycle management for optional cache access."""

from __future__ import annotations

import logging

from redis.asyncio import Redis
from redis.exceptions import RedisError

from api.config import settings

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

    async def _close_client(self, client: Redis) -> None:
        """Close a Redis client across redis-py versions."""

        close = getattr(client, "aclose", None)
        if close is not None:
            await close()
            return

        await client.close()


redis_cache_service = RedisCacheService()
