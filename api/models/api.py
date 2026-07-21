"""Shared public API response schemas."""

from pydantic import BaseModel, Field


class ApiErrorDetail(BaseModel):
    """A safe, machine-readable API error description."""

    code: str = Field(examples=["invalid_image"])
    message: str = Field(examples=["Uploaded image could not be decoded."])


class ApiErrorResponse(BaseModel):
    """Standard error envelope returned by all API routes."""

    error: ApiErrorDetail
