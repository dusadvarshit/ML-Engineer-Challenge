"""Minimal FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from api.config import settings
from api.metrics import instrument_http_request, metrics_response, set_dependency_status
from api.middleware.request_logging import log_http_exchange
from api.routers.classification import router as classification_router
from api.routers.object_detection import router as object_detection_router
from api.services.cache_service import redis_cache_service
from api.services.object_detection.yolo_service import yolo_prediction_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load API resources before the application starts serving traffic."""

    set_dependency_status(dependency='postgres', is_available=True)
    await redis_cache_service.startup()
    set_dependency_status(
        dependency='redis',
        is_available=redis_cache_service.is_available,
    )
    yolo_prediction_service.load()
    set_dependency_status(dependency='yolo_model', is_available=True)
    try:
        yield
    finally:
        set_dependency_status(dependency='yolo_model', is_available=False)
        await redis_cache_service.shutdown()
        set_dependency_status(dependency='redis', is_available=False)
        set_dependency_status(dependency='postgres', is_available=False)


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
)


@app.middleware('http')
async def prometheus_metrics_middleware(request: Request, call_next) -> Response:
    """Collect Prometheus request metrics for each HTTP exchange."""

    return await instrument_http_request(request, call_next)


@app.middleware('http')
async def request_logging_middleware(request: Request, call_next) -> Response:
    """Queue request/response logs for asynchronous persistence."""

    return await log_http_exchange(request, call_next)


app.include_router(object_detection_router, prefix=settings.API_PREFIX)
app.include_router(classification_router, prefix=settings.API_PREFIX)


@app.get('/')
def read_root() -> dict[str, str]:
    """Return a simple API identification payload."""

    return {'message': settings.API_TITLE}


@app.get('/health')
def health_check() -> dict[str, str]:
    """Return a basic health status for the API."""

    return {'status': 'ok'}


@app.get(f'{settings.API_PREFIX}/metrics', include_in_schema=False)
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return metrics_response()
