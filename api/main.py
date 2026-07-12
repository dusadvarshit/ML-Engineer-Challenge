"""Minimal FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from api.config import settings
from api.routers.classification import router as classification_router
from api.routers.object_detection import router as object_detection_router
from api.services.object_detection.yolo_service import yolo_prediction_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load API resources before the application starts serving traffic."""

    yolo_prediction_service.load()
    yield


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    lifespan=lifespan,
)
app.include_router(object_detection_router, prefix=settings.API_PREFIX)
app.include_router(classification_router, prefix=settings.API_PREFIX)


@app.get("/")
def read_root() -> dict[str, str]:
    """Return a simple API identification payload."""

    return {"message": settings.API_TITLE}


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a basic health status for the API."""

    return {"status": "ok"}
