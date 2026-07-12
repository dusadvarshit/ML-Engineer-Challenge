"""Minimal FastAPI application entrypoint."""

from fastapi import FastAPI

from api.config import settings


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": settings.API_TITLE}


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
