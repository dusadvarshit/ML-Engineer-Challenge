"""Global exception handlers for stable public API errors."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    """Build the one supported public error envelope."""

    return {"error": {"code": code, "message": message}}


async def http_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Serialize expected client and dependency errors safely."""

    if not isinstance(exc, HTTPException):
        raise TypeError(
            "HTTP exception handler received an unexpected exception."
        )

    if isinstance(exc.detail, dict):
        code = str(exc.detail.get("code", "request_error"))
        message = str(
            exc.detail.get("message", "Request could not be completed.")
        )
    else:
        code = _status_code_name(exc.status_code)
        message = str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code, content=_error_payload(code, message)
    )


async def validation_exception_handler(
    _: Request, exc: Exception
) -> JSONResponse:
    """Keep framework validation details out of the public contract."""

    if not isinstance(exc, RequestValidationError):
        raise TypeError(
            "Validation exception handler received an unexpected exception."
        )

    first_error: dict[str, Any] | None = next(iter(exc.errors()), None)
    message = "Request payload is invalid."
    if first_error and first_error.get("type") == "missing":
        message = "A required request field is missing."
    return JSONResponse(
        status_code=422, content=_error_payload("validation_error", message)
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Avoid exposing stack traces or internal exceptions to callers."""

    logger.exception(
        "Unhandled API error for %s %s", request.method, request.url.path
    )
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            "internal_error", "An internal server error occurred."
        ),
    )


def _status_code_name(status_code: int) -> str:
    """Return a compact, stable code for common HTTP failures."""

    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        413: "payload_too_large",
        415: "unsupported_media_type",
        422: "validation_error",
        429: "rate_limited",
        500: "inference_failed",
        503: "service_unavailable",
    }.get(status_code, "request_error")
