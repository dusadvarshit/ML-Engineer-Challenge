"""Unit tests for the FastAPI entrypoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY

import api.main as main_module
from api.config import settings
from api.routers import object_detection as router_module

pytestmark = pytest.mark.unit


class StubPredictionService:
    """Small prediction stub for request logging tests."""

    def get_model_version(self) -> str:
        return "yolov8n"

    def predict(self, _: bytes) -> list[object]:
        return []


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

    monkeypatch.setattr(
        main_module.redis_cache_service, "startup", startup or noop_async
    )
    monkeypatch.setattr(
        main_module.redis_cache_service, "shutdown", shutdown or noop_async
    )
    monkeypatch.setattr(
        main_module.yolo_prediction_service, "load", load or (lambda: None)
    )


def test_read_root_returns_api_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """The root route should expose the configured API title."""

    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setattr(
        settings,
        "API_KEYS",
        '{"test-client":{"api_key":"test-api-key","tier":"standard"}}',
    )

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


def test_models_endpoint_reports_public_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model discovery must not expose artifact paths to callers."""

    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setattr(
        main_module,
        "get_object_detection_model_metadata",
        lambda: [],
    )

    with TestClient(main_module.app) as client:
        response = client.get(f"{settings.API_PREFIX}/models")

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_readiness_reports_degraded_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Readiness is dependency-aware while /health remains liveness-only."""

    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setattr(main_module, "_database_is_available", lambda: False)
    monkeypatch.setattr(
        main_module.redis_cache_service, "_is_available", False
    )
    monkeypatch.setattr(main_module.yolo_prediction_service, "_model", None)

    with TestClient(main_module.app) as client:
        response = client.get(f"{settings.API_PREFIX}/ready")

    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "dependencies": {
            "postgres": False,
            "redis": False,
            "yolo_model": False,
        },
    }


def test_metrics_endpoint_exposes_prometheus_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The metrics route should expose Prometheus text format."""

    _patch_lifespan_dependencies(monkeypatch)

    with TestClient(main_module.app) as client:
        response = client.get(f"{settings.API_PREFIX}/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(CONTENT_TYPE_LATEST)
    assert "ml_api_http_requests_total" in response.text
    assert "ml_model_inference_requests_total" in response.text


def test_health_request_increments_http_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request middleware should publish counters and latency for health checks."""

    _patch_lifespan_dependencies(monkeypatch)

    request_labels = {"method": "GET", "path": "/health", "status": "200"}
    duration_labels = {"method": "GET", "path": "/health"}
    before_requests = (
        REGISTRY.get_sample_value("ml_api_http_requests_total", request_labels)
        or 0.0
    )
    before_duration_count = (
        REGISTRY.get_sample_value(
            "ml_api_http_request_duration_seconds_count", duration_labels
        )
        or 0.0
    )

    with TestClient(main_module.app) as client:
        response = client.get("/health")

    after_requests = (
        REGISTRY.get_sample_value("ml_api_http_requests_total", request_labels)
        or 0.0
    )
    after_duration_count = (
        REGISTRY.get_sample_value(
            "ml_api_http_request_duration_seconds_count", duration_labels
        )
        or 0.0
    )

    assert response.status_code == 200
    assert after_requests == before_requests + 1
    assert after_duration_count == before_duration_count + 1


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


def test_health_request_enqueues_request_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request middleware should enqueue one structured request log."""

    _patch_lifespan_dependencies(monkeypatch)
    captured_payloads: list[dict[str, object]] = []

    monkeypatch.setattr(
        "api.middleware.request_logging.persist_request_log.delay",
        lambda payload: captured_payloads.append(payload),
    )

    with TestClient(main_module.app) as client:
        response = client.get(
            "/health",
            headers={
                "X-Request-Id": "req-123",
                "X-User-Id": "alice",
                "User-Agent": "pytest-client",
            },
        )

    assert response.status_code == 200
    assert response.headers["X-Request-Id"] == "req-123"
    assert len(captured_payloads) == 1

    payload = captured_payloads[0]
    assert payload["request_id"] == "req-123"
    assert payload["method"] == "GET"
    assert payload["path"] == "/health"
    assert payload["route_path"] == "/health"
    assert payload["status_code"] == 200
    assert payload["user_id"] == "alice"
    assert payload["response_payload"] == {"status": "ok"}
    assert payload["latency_ms"] >= 0


def test_detect_request_logs_form_fields_and_file_metadata(
    monkeypatch: pytest.MonkeyPatch,
    sample_image_bytes: bytes,
) -> None:
    """Multipart logging should capture safe form metadata rather than file bytes."""

    _patch_lifespan_dependencies(monkeypatch)
    monkeypatch.setattr(
        settings,
        "API_KEYS",
        '{"test-client":{"api_key":"test-api-key","tier":"standard"}}',
    )
    captured_payloads: list[dict[str, object]] = []

    async def fake_get_detection(*_args, **_kwargs):
        return None

    async def fake_set_detection(*_args, **_kwargs):
        return True

    monkeypatch.setattr(
        "api.middleware.request_logging.persist_request_log.delay",
        lambda payload: captured_payloads.append(payload),
    )
    monkeypatch.setattr(
        router_module,
        "get_object_detection_service",
        lambda _model: StubPredictionService(),
    )
    monkeypatch.setattr(
        main_module.redis_cache_service, "get_detection", fake_get_detection
    )
    monkeypatch.setattr(
        main_module.redis_cache_service, "set_detection", fake_set_detection
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            f"{settings.API_PREFIX}/detect",
            data={"model": "yolov8n"},
            files={"file": ("image.png", sample_image_bytes, "image/png")},
            headers={settings.API_KEY_HEADER_NAME: "test-api-key"},
        )

    assert response.status_code == 200
    assert len(captured_payloads) == 1
    request_payload = captured_payloads[0]["request_payload"]
    assert request_payload["form"]["model"] == "yolov8n"
    assert request_payload["files"]["file"] == {
        "filename": "image.png",
        "content_type": "image/png",
    }
