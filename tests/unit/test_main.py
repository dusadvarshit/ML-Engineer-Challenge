"""Unit tests for the FastAPI entrypoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as main_module
from api.config import settings

pytestmark = pytest.mark.unit


def _patch_lifespan_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    load=None,
    startup=None,
    shutdown=None,
) -> None:
    """Patch startup dependencies so app tests stay deterministic."""

    async def noop_async() -> None:
        return None

    monkeypatch.setattr(main_module.redis_cache_service, "startup", startup or noop_async)
    monkeypatch.setattr(main_module.redis_cache_service, "shutdown", shutdown or noop_async)
    monkeypatch.setattr(main_module.yolo_prediction_service, "load", load or (lambda: None))


def test_read_root_returns_api_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root route should expose the configured API title."""

    _patch_lifespan_dependencies(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": settings.API_TITLE}


def test_health_check_returns_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """The health route should report a healthy status."""

    _patch_lifespan_dependencies(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_lifespan_initializes_and_closes_dependencies_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """App startup should initialize Redis and YOLO, then shut Redis down once."""

    calls = {"startup": 0, "load": 0, "shutdown": 0}

    async def fake_startup() -> None:
        calls["startup"] += 1

    def fake_load() -> None:
        calls["load"] += 1

    async def fake_shutdown() -> None:
        calls["shutdown"] += 1

    _patch_lifespan_dependencies(
        monkeypatch,
        load=fake_load,
        startup=fake_startup,
        shutdown=fake_shutdown,
    )

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert calls == {"startup": 1, "load": 1, "shutdown": 1}
