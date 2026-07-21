"""Minimal FastAPI application entrypoint."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from api.config import settings
from api.database import engine
from api.errors import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from api.metrics import instrument_http_request, metrics_response, set_dependency_status
from api.models.model_metadata import ModelListResponse, ReadinessResponse
from api.middleware.request_logging import log_http_exchange
from api.routers.classification import router as classification_router
from api.routers.object_detection import router as object_detection_router
from api.services.cache_service import redis_cache_service
from api.services.object_detection.yolo_service import yolo_prediction_service
from api.services.object_detection.registry import get_model_metadata
from api.services.classification import classification_prediction_service


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
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)


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
    """Liveness probe: the API process is able to serve HTTP."""

    return {'status': 'ok'}


@app.get(f'{settings.API_PREFIX}/ready', response_model=ReadinessResponse)
def readiness_check() -> Response:
    """Readiness probe for required serving dependencies."""

    dependencies = {
        'postgres': _database_is_available(),
        'redis': redis_cache_service.is_available,
        'yolo_model': getattr(yolo_prediction_service, '_model', None) is not None,
        'classification_model': classification_prediction_service.is_available,
    }
    ready = all(dependencies.values())
    response = ReadinessResponse(
        status='ready' if ready else 'degraded', dependencies=dependencies
    )
    return JSONResponse(status_code=200 if ready else 503, content=response.model_dump())


@app.get(f'{settings.API_PREFIX}/models', response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    """Return model artifact availability and loaded readiness."""

    return ModelListResponse(models=get_model_metadata())


@app.get(f'{settings.API_PREFIX}/metrics', include_in_schema=False)
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics for scraping."""

    return metrics_response()


def _database_is_available() -> bool:
    """Check PostgreSQL only for readiness; liveness remains dependency-free."""

    try:
        with engine.connect() as connection:
            connection.execute(text('SELECT 1'))
    except SQLAlchemyError:
        return False
    return True
