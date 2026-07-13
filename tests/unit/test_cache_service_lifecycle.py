"""Additional unit tests for Redis cache lifecycle management."""

from __future__ import annotations

import asyncio

import pytest

from api.services.cache_service import RedisCacheService

pytestmark = pytest.mark.unit


class HealthyRedisClient:
    """Async Redis stub that succeeds and tracks shutdown."""

    def __init__(self) -> None:
        self.pinged = False
        self.closed = False

    async def ping(self) -> None:
        self.pinged = True

    async def aclose(self) -> None:
        self.closed = True


class LegacyRedisClient:
    """Async Redis stub that only exposes the legacy close method."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_startup_marks_service_available_when_ping_succeeds(mocker) -> None:
    """Successful Redis startup should retain the live client."""

    client = HealthyRedisClient()
    mocker.patch("api.services.cache_service.Redis.from_url", return_value=client)

    service = RedisCacheService()

    asyncio.run(service.startup())

    assert service.client is client
    assert service.is_available is True
    assert client.pinged is True


def test_shutdown_closes_active_client_and_clears_state() -> None:
    """Shutdown should close the active async Redis client and clear availability."""

    service = RedisCacheService()
    client = HealthyRedisClient()
    service._client = client
    service._is_available = True

    asyncio.run(service.shutdown())

    assert client.closed is True
    assert service.client is None
    assert service.is_available is False


def test_close_client_falls_back_to_legacy_close() -> None:
    """Older redis-py clients without aclose should still be supported."""

    service = RedisCacheService()
    client = LegacyRedisClient()

    asyncio.run(service._close_client(client))

    assert client.closed is True
