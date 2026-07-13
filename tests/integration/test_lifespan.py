"""Integration tests for FastAPI lifespan behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import api.main as main_module

pytestmark = pytest.mark.integration


def test_app_startup_succeeds_when_dependencies_are_mocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mounted app should serve requests when startup dependencies are mocked."""

    calls = {"startup": 0, "load": 0, "shutdown": 0}

    async def fake_startup() -> None:
        calls["startup"] += 1

    def fake_load() -> None:
        calls["load"] += 1

    async def fake_shutdown() -> None:
        calls["shutdown"] += 1

    monkeypatch.setattr(main_module.redis_cache_service, "startup", fake_startup)
    monkeypatch.setattr(main_module.redis_cache_service, "shutdown", fake_shutdown)
    monkeypatch.setattr(main_module.yolo_prediction_service, "load", fake_load)

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert calls == {"startup": 1, "load": 1, "shutdown": 1}


def test_startup_failure_from_cache_startup_bubbles_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis startup failures should be visible during app startup."""

    async def raise_startup_error() -> None:
        raise RuntimeError("redis startup failed")

    async def fake_shutdown() -> None:
        return None

    monkeypatch.setattr(main_module.redis_cache_service, "startup", raise_startup_error)
    monkeypatch.setattr(main_module.redis_cache_service, "shutdown", fake_shutdown)
    monkeypatch.setattr(main_module.yolo_prediction_service, "load", lambda: None)

    with pytest.raises(RuntimeError, match="redis startup failed"):
        with TestClient(main_module.app):
            pass
