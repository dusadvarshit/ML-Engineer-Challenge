"""Unit tests for Prometheus metrics helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import Response

import api.metrics as metrics_module

pytestmark = pytest.mark.unit


class FakeMetricChild:
    """Record metric operations performed on one label set."""

    def __init__(self) -> None:
        self.inc_calls: list[float] = []
        self.observe_calls: list[float] = []
        self.set_calls: list[int] = []

    def inc(self, amount: float = 1) -> None:
        self.inc_calls.append(amount)

    def observe(self, value: float) -> None:
        self.observe_calls.append(value)

    def set(self, value: int) -> None:
        self.set_calls.append(value)


class FakeMetric:
    """Record labels requested and expose per-label metric children."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, str], FakeMetricChild]] = []

    def labels(self, **labels: str) -> FakeMetricChild:
        child = FakeMetricChild()
        self.calls.append((labels, child))
        return child


class FakeURL:
    """Minimal request URL object."""

    def __init__(self, path: str) -> None:
        self.path = path


class FakeRequest:
    """Minimal request object for middleware tests."""

    def __init__(self, *, method: str = 'GET', path: str = '/health', route_path: str | None = None) -> None:
        self.method = method
        self.url = FakeURL(path)
        route = SimpleNamespace(path=route_path) if route_path is not None else None
        self.scope = {'route': route}


def test_metrics_response_returns_prometheus_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """The metrics endpoint should expose the generated Prometheus payload."""

    monkeypatch.setattr(metrics_module, 'generate_latest', lambda: b'metric 1\n')

    response = metrics_module.metrics_response()

    assert response.body == b'metric 1\n'
    assert response.media_type == metrics_module.CONTENT_TYPE_LATEST


def test_instrument_http_request_records_success_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successful requests should increment request counters and latency histograms."""

    request_counter = FakeMetric()
    duration_histogram = FakeMetric()
    monkeypatch.setattr(metrics_module, 'HTTP_REQUESTS_TOTAL', request_counter)
    monkeypatch.setattr(metrics_module, 'HTTP_REQUEST_DURATION_SECONDS', duration_histogram)
    monkeypatch.setattr(metrics_module, 'perf_counter', iter([10.0, 10.25]).__next__)

    async def call_next(_request):
        return Response(status_code=204)

    response = asyncio.run(
        metrics_module.instrument_http_request(
            FakeRequest(path='/health', route_path='/health'),
            call_next,
        )
    )

    assert response.status_code == 204
    assert request_counter.calls[0][0] == {'method': 'GET', 'path': '/health', 'status': '204'}
    assert request_counter.calls[0][1].inc_calls == [1]
    assert duration_histogram.calls[0][0] == {'method': 'GET', 'path': '/health'}
    assert duration_histogram.calls[0][1].observe_calls == [0.25]


def test_instrument_http_request_records_exception_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unhandled request errors should increment both exception and request metrics."""

    request_counter = FakeMetric()
    exception_counter = FakeMetric()
    duration_histogram = FakeMetric()
    monkeypatch.setattr(metrics_module, 'HTTP_REQUESTS_TOTAL', request_counter)
    monkeypatch.setattr(metrics_module, 'HTTP_REQUEST_EXCEPTIONS_TOTAL', exception_counter)
    monkeypatch.setattr(metrics_module, 'HTTP_REQUEST_DURATION_SECONDS', duration_histogram)
    monkeypatch.setattr(metrics_module, 'perf_counter', iter([20.0, 20.5]).__next__)

    async def call_next(_request):
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        asyncio.run(
            metrics_module.instrument_http_request(
                FakeRequest(path='/fallback', route_path=None),
                call_next,
            )
        )

    assert exception_counter.calls[0][0] == {
        'method': 'GET',
        'path': '/fallback',
        'exception_type': 'RuntimeError',
    }
    assert exception_counter.calls[0][1].inc_calls == [1]
    assert request_counter.calls[0][0] == {'method': 'GET', 'path': '/fallback', 'status': '500'}
    assert request_counter.calls[0][1].inc_calls == [1]
    assert duration_histogram.calls[0][1].observe_calls == [0.5]


def test_observe_inference_tracks_volume_and_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inference observation should record both throughput and latency."""

    request_counter = FakeMetric()
    duration_histogram = FakeMetric()
    monkeypatch.setattr(metrics_module, 'MODEL_INFERENCE_REQUESTS_TOTAL', request_counter)
    monkeypatch.setattr(metrics_module, 'MODEL_INFERENCE_DURATION_SECONDS', duration_histogram)

    metrics_module.observe_inference(
        task='detect',
        model='yolov8n',
        outcome='success',
        duration_seconds=0.123,
        image_count=3,
    )

    assert request_counter.calls[0][0] == {
        'task': 'detect',
        'model': 'yolov8n',
        'outcome': 'success',
    }
    assert request_counter.calls[0][1].inc_calls == [3]
    assert duration_histogram.calls[0][1].observe_calls == [0.123]


def test_observe_batch_request_tracks_count_and_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch observation should record the request and batch size."""

    batch_counter = FakeMetric()
    batch_size = FakeMetric()
    monkeypatch.setattr(metrics_module, 'BATCH_REQUESTS_TOTAL', batch_counter)
    monkeypatch.setattr(metrics_module, 'BATCH_SIZE', batch_size)

    metrics_module.observe_batch_request(task='detect', batch_size=4, outcome='success')

    assert batch_counter.calls[0][0] == {'task': 'detect', 'outcome': 'success'}
    assert batch_counter.calls[0][1].inc_calls == [1]
    assert batch_size.calls[0][0] == {'task': 'detect'}
    assert batch_size.calls[0][1].observe_calls == [4]


def test_set_dependency_status_updates_gauge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dependency availability should be exported as 1 or 0."""

    dependency_gauge = FakeMetric()
    monkeypatch.setattr(metrics_module, 'DEPENDENCY_UP', dependency_gauge)

    metrics_module.set_dependency_status(dependency='redis', is_available=True)
    metrics_module.set_dependency_status(dependency='redis', is_available=False)

    assert dependency_gauge.calls[0][0] == {'dependency': 'redis'}
    assert dependency_gauge.calls[0][1].set_calls == [1]
    assert dependency_gauge.calls[1][1].set_calls == [0]


def test_resolve_path_template_prefers_route_path() -> None:
    """Route templates should be used when available to keep label cardinality low."""

    path = metrics_module._resolve_path_template(FakeRequest(path='/api/v1/detect', route_path='/api/v1/detect'))

    assert path == '/api/v1/detect'


def test_resolve_path_template_falls_back_to_raw_path() -> None:
    """Raw URL paths should be used when no route template exists."""

    path = metrics_module._resolve_path_template(FakeRequest(path='/raw-path', route_path=None))

    assert path == '/raw-path'
