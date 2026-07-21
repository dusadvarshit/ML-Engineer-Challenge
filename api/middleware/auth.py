"""API-key authentication and Redis-backed rate limiting for API routes."""

from __future__ import annotations

import hashlib
import hmac
import logging
from dataclasses import dataclass
from time import time

from fastapi import HTTPException, Request, status
from redis.exceptions import RedisError

from api.config import ApiClient, settings
from api.services.cache_service import redis_cache_service

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AuthenticatedClient:
    """The authenticated, non-secret identity available to route handlers."""

    client_id: str
    tier: str


async def require_api_key(request: Request) -> AuthenticatedClient:
    """Authenticate an API client and enforce its configured request limit."""

    api_key = request.headers.get(settings.API_KEY_HEADER_NAME)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
        )

    try:
        configured_clients = settings.get_api_clients()
    except ValueError as exc:
        logger.error("Invalid API key configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is unavailable.",
        ) from exc

    client = _find_client(api_key, configured_clients)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key."
        )

    authenticated_client = AuthenticatedClient(
        client_id=client.client_id, tier=client.tier
    )
    await _enforce_rate_limit(authenticated_client)
    request.state.api_client = authenticated_client
    return authenticated_client


def _find_client(
    api_key: str, configured_clients: tuple[ApiClient, ...]
) -> ApiClient | None:
    """Constant-time compare a presented key against configured client keys."""

    matched_client: ApiClient | None = None
    for client in configured_clients:
        if hmac.compare_digest(api_key, client.api_key):
            matched_client = client
    return matched_client


async def _enforce_rate_limit(client: AuthenticatedClient) -> None:
    """Increment the current Redis minute bucket and reject excess requests."""

    try:
        limits = settings.get_rate_limits()
    except (TypeError, ValueError) as exc:
        logger.error("Invalid rate-limit configuration: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is unavailable.",
        ) from exc

    limit = limits.get(
        client.tier, limits.get("standard", settings.RATE_LIMIT_PER_MINUTE)
    )
    redis_client = redis_cache_service.client
    if redis_client is None:
        logger.warning(
            "Redis unavailable; skipping API rate limit for client %s.",
            client.client_id,
        )
        return

    current_minute = int(time() // 60)
    client_digest = hashlib.sha256(client.client_id.encode("utf-8")).hexdigest()
    key = f"ratelimit:api:{client_digest}:{current_minute}"
    increment = getattr(redis_client, "incr", None)
    expire = getattr(redis_client, "expire", None)
    if increment is None or expire is None:
        logger.debug("Redis client does not support rate-limit operations.")
        return
    try:
        count = await increment(key)
        if count == 1:
            await expire(key, 60)
    except (OSError, RedisError) as exc:
        logger.warning("Redis rate-limit check failed: %s", exc)
        return

    if count > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
            headers={"Retry-After": "60"},
        )
