"""Schemas for model discovery and readiness responses."""

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Public operational metadata for one inference model."""

    name: str
    task: str = "detection"
    version: str | None = None
    artifact_available: bool
    loaded: bool
    ready: bool


class ModelListResponse(BaseModel):
    """Model discovery response without local filesystem details."""

    models: list[ModelMetadata]


class ReadinessResponse(BaseModel):
    """Dependency-aware readiness status."""

    status: str = Field(examples=["ready"])
    dependencies: dict[str, bool]
