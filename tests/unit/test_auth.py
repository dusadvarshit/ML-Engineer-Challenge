"""Tests for API-key authentication and rate limiting."""

from __future__ import annotations

import pytest

from api.config import settings
from api.middleware import auth as auth_module

pytestmark = pytest.mark.unit


def test_protected_route_rejects_missing_api_key(client) -> None:
    """Inference routes should require an environment-provisioned API key."""

    del client.headers[settings.API_KEY_HEADER_NAME]
    response = client.post(f"{settings.API_PREFIX}/detect")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}


def test_protected_route_rejects_invalid_api_key(client) -> None:
    """Unknown credentials should not disclose configured client identities."""

    response = client.post(
        f"{settings.API_PREFIX}/detect",
        headers={settings.API_KEY_HEADER_NAME: "not-a-valid-key"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid API key."}


def test_api_key_is_not_required_for_health_or_metrics(client) -> None:
    """Infrastructure probes remain available without client credentials."""

    del client.headers[settings.API_KEY_HEADER_NAME]

    assert client.get("/health").status_code == 200
    assert client.get(f"{settings.API_PREFIX}/metrics").status_code == 200


def test_api_key_rate_limit_uses_redis_counter(
    client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Configured API-client tiers should reject requests above their Redis limit."""

    class FakeRateLimitRedis:
        def __init__(self) -> None:
            self.count = 0

        async def incr(self, _key: str) -> int:
            self.count += 1
            return self.count

        async def expire(self, _key: str, _seconds: int) -> bool:
            return True

    fake_redis = FakeRateLimitRedis()
    monkeypatch.setattr(settings, "RATE_LIMITS_PER_MINUTE", "standard:1")
    monkeypatch.setattr(auth_module.redis_cache_service, "_client", fake_redis)
    monkeypatch.setattr(auth_module.redis_cache_service, "_is_available", True)

    assert client.post(f"{settings.API_PREFIX}/detect").status_code == 422
    response = client.post(f"{settings.API_PREFIX}/detect")

    assert response.status_code == 429
    assert response.headers["retry-after"] == "60"
