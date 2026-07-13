"""Prometheus metrics helpers for API and inference observability."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

HTTP_REQUESTS_TOTAL = Counter(
    'ml_api_http_requests_total',
    'Total HTTP requests served by the API.',
    ('method', 'path', 'status'),
)
HTTP_REQUEST_EXCEPTIONS_TOTAL = Counter(
    'ml_api_http_request_exceptions_total',
    'Unhandled HTTP request exceptions raised by the API.',
    ('method', 'path', 'exception_type'),
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'ml_api_http_request_duration_seconds',
    'End-to-end HTTP request latency in seconds.',
    ('method', 'path'),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
MODEL_INFERENCE_REQUESTS_TOTAL = Counter(
    'ml_model_inference_requests_total',
    'Total model inference operations performed by the API.',
    ('task', 'model', 'outcome'),
)
MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    'ml_model_inference_duration_seconds',
    'Time spent performing model inference in seconds.',
    ('task', 'model', 'outcome'),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
BATCH_REQUESTS_TOTAL = Counter(
    'ml_batch_requests_total',
    'Total batch inference jobs handled by the API.',
    ('task', 'outcome'),
)
BATCH_SIZE = Histogram(
    'ml_batch_size',
    'Number of images processed in each batch inference job.',
    ('task',),
    buckets=(1, 2, 4, 8, 16, 32),
)
CACHE_OPERATIONS_TOTAL = Counter(
    'ml_cache_operations_total',
    'Total cache lookups and writes performed by the API.',
    ('cache', 'operation', 'outcome'),
)
DEPENDENCY_UP = Gauge(
    'ml_dependency_up',
    'Whether a serving dependency is currently available.',
    ('dependency',),
)


def metrics_response() -> Response:
    """Return the current Prometheus exposition payload."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def instrument_http_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Collect HTTP request metrics for every API request."""

    method = request.method
    started_at = perf_counter()

    try:
        response = await call_next(request)
    except Exception as exc:
        path = _resolve_path_template(request)
        HTTP_REQUEST_EXCEPTIONS_TOTAL.labels(
            method=method,
            path=path,
            exception_type=type(exc).__name__,
        ).inc()
        HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status='500').inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
            perf_counter() - started_at
        )
        raise

    path = _resolve_path_template(request)
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        path=path,
        status=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, path=path).observe(
        perf_counter() - started_at
    )
    return response


def observe_inference(
    *,
    task: str,
    model: str,
    outcome: str,
    duration_seconds: float,
    image_count: int = 1,
) -> None:
    """Record model inference counters and latency."""

    MODEL_INFERENCE_REQUESTS_TOTAL.labels(
        task=task,
        model=model,
        outcome=outcome,
    ).inc(image_count)
    MODEL_INFERENCE_DURATION_SECONDS.labels(
        task=task,
        model=model,
        outcome=outcome,
    ).observe(duration_seconds)


def observe_batch_request(*, task: str, batch_size: int, outcome: str) -> None:
    """Record batch job counters and batch-size distribution."""

    BATCH_REQUESTS_TOTAL.labels(task=task, outcome=outcome).inc()
    BATCH_SIZE.labels(task=task).observe(batch_size)


def observe_cache_operation(*, cache: str, operation: str, outcome: str) -> None:
    """Record cache lookup and write outcomes."""

    CACHE_OPERATIONS_TOTAL.labels(
        cache=cache,
        operation=operation,
        outcome=outcome,
    ).inc()


def set_dependency_status(*, dependency: str, is_available: bool) -> None:
    """Update a dependency availability gauge."""

    DEPENDENCY_UP.labels(dependency=dependency).set(1 if is_available else 0)


def _resolve_path_template(request: Request) -> str:
    """Return a low-cardinality path label for one request."""

    route = request.scope.get('route')
    route_path = getattr(route, 'path', None)
    if route_path:
        return route_path
    return request.url.path
