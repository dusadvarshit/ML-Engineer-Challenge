"""HTTP middleware for asynchronous request logging."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any
from urllib.parse import parse_qs
from uuid import uuid4

from fastapi import Request, Response
from starlette.concurrency import iterate_in_threadpool
from starlette.datastructures import FormData, UploadFile

from api.config import settings
from api.worker import persist_request_log


async def log_http_exchange(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Enqueue one background task that persists request and response metadata."""

    if not settings.REQUEST_LOGGING_ENABLED:
        return await call_next(request)

    request_id = request.headers.get(settings.REQUEST_ID_HEADER_NAME) or str(
        uuid4()
    )
    request.state.request_id = request_id

    raw_body = await request.body()
    _reset_request_body(request, raw_body)
    request_payload = await _build_request_payload(request, raw_body)
    _reset_request_body(request, raw_body)

    started_at = perf_counter()
    method = request.method
    path = request.url.path

    try:
        response = await call_next(request)
    except Exception as exc:
        _enqueue_request_log(
            {
                "request_id": request_id,
                "method": method,
                "path": path,
                "route_path": _resolve_route_path(request),
                "status_code": 500,
                "latency_ms": round((perf_counter() - started_at) * 1000, 3),
                "user_id": _resolve_user_id(request),
                "client_ip": _resolve_client_ip(request),
                "user_agent": request.headers.get("user-agent"),
                "request_content_type": request.headers.get("content-type"),
                "response_content_type": "application/json",
                "request_payload": request_payload,
                "response_payload": {
                    "detail": _truncate_text(str(exc)),
                    "error_type": type(exc).__name__,
                },
                "error_message": _truncate_text(str(exc)),
            }
        )
        raise

    response_body = await _capture_response_body(response)
    if settings.REQUEST_ID_HEADER_NAME not in response.headers:
        response.headers[settings.REQUEST_ID_HEADER_NAME] = request_id
    _enqueue_request_log(
        {
            "request_id": request_id,
            "method": method,
            "path": path,
            "route_path": _resolve_route_path(request),
            "status_code": response.status_code,
            "latency_ms": round((perf_counter() - started_at) * 1000, 3),
            "user_id": _resolve_user_id(request),
            "client_ip": _resolve_client_ip(request),
            "user_agent": request.headers.get("user-agent"),
            "request_content_type": request.headers.get("content-type"),
            "response_content_type": response.headers.get("content-type"),
            "request_payload": request_payload,
            "response_payload": _serialize_response_payload(
                body=response_body,
                content_type=response.headers.get("content-type"),
            ),
            "error_message": (
                None
                if response.status_code < 400
                else _extract_error_message(response_body)
            ),
        }
    )
    return response


def _enqueue_request_log(payload: dict[str, Any]) -> None:
    """Fire-and-forget one Celery task without failing the request path."""

    try:
        persist_request_log.delay(payload)
    except Exception:
        return


def _reset_request_body(request: Request, raw_body: bytes) -> None:
    """Replace the ASGI receive hook so downstream handlers can re-read the body."""

    async def receive() -> dict[str, Any]:
        return {"type": "http.request", "body": raw_body, "more_body": False}

    request._receive = receive


async def _build_request_payload(
    request: Request, raw_body: bytes
) -> dict[str, Any]:
    """Serialize one inbound request into a JSON-safe structure."""

    payload: dict[str, Any] = {
        "path_params": _sanitize_value(dict(request.path_params)),
        "query_params": _sanitize_value(dict(request.query_params)),
    }
    content_type = request.headers.get("content-type", "")
    if not raw_body:
        return payload

    if content_type.startswith("application/json"):
        payload["body"] = _parse_json_body(raw_body)
        return payload

    if content_type.startswith("application/x-www-form-urlencoded"):
        payload["form"] = _flatten_query_values(
            parse_qs(raw_body.decode("utf-8"), keep_blank_values=True)
        )
        return payload

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        payload.update(_serialize_form_data(form))
        return payload

    payload["body_preview"] = _truncate_text(
        raw_body.decode("utf-8", errors="replace")
    )
    return payload


def _parse_json_body(raw_body: bytes) -> Any:
    """Parse one JSON request body into a safe, bounded structure."""

    try:
        return _sanitize_value(json.loads(raw_body))
    except json.JSONDecodeError:
        return {
            "body_preview": _truncate_text(
                raw_body.decode("utf-8", errors="replace")
            )
        }


def _flatten_query_values(values: dict[str, list[str]]) -> dict[str, Any]:
    """Collapse single-item query string values for easier reading."""

    flattened: dict[str, Any] = {}
    for key, items in values.items():
        if len(items) == 1:
            flattened[key] = _sanitize_value(items[0])
            continue
        flattened[key] = _sanitize_value(items)
    return flattened


def _serialize_form_data(form: FormData) -> dict[str, Any]:
    """Split multipart fields into JSON fields and file metadata."""

    fields: dict[str, Any] = {}
    files: dict[str, Any] = {}

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            _append_form_value(
                files,
                key,
                {
                    "filename": value.filename,
                    "content_type": value.content_type,
                },
            )
            continue

        _append_form_value(fields, key, _sanitize_value(value))

    payload: dict[str, Any] = {}
    if fields:
        payload["form"] = fields
    if files:
        payload["files"] = files
    return payload


def _append_form_value(target: dict[str, Any], key: str, value: Any) -> None:
    """Store one form value while preserving repeated keys."""

    if key not in target:
        target[key] = value
        return

    existing = target[key]
    if isinstance(existing, list):
        existing.append(value)
        return

    target[key] = [existing, value]


async def _capture_response_body(response: Response) -> bytes:
    """Read one response body and restore it for the client."""

    body = getattr(response, "body", None)
    if body is not None:
        return body

    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        return b""

    chunks: list[bytes] = []
    async for chunk in body_iterator:
        chunks.append(chunk)

    setattr(response, "body_iterator", iterate_in_threadpool(iter(chunks)))
    return b"".join(chunks)


def _serialize_response_payload(
    *, body: bytes, content_type: str | None
) -> dict[str, Any] | None:
    """Serialize one response body into a JSON-safe structure."""

    if not body:
        return None

    normalized_content_type = (
        (content_type or "").split(";", 1)[0].strip().lower()
    )
    if normalized_content_type == "application/json":
        try:
            return _sanitize_value(json.loads(body))
        except json.JSONDecodeError:
            return {
                "body_preview": _truncate_text(
                    body.decode("utf-8", errors="replace")
                )
            }

    if normalized_content_type.startswith("text/"):
        return {
            "body_preview": _truncate_text(
                body.decode("utf-8", errors="replace")
            )
        }

    return {"size_bytes": len(body)}


def _extract_error_message(response_body: bytes) -> str | None:
    """Pull one concise error summary from the serialized response body."""

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError:
        text = response_body.decode("utf-8", errors="replace").strip()
        return _truncate_text(text) or None

    if isinstance(payload, dict) and "detail" in payload:
        return _truncate_text(str(payload["detail"]))
    return None


def _resolve_user_id(request: Request) -> str:
    """Resolve one user identifier from request headers."""

    authenticated_client = getattr(request.state, "api_client", None)
    if authenticated_client is not None:
        return authenticated_client.client_id
    return request.headers.get(settings.REQUEST_USER_HEADER_NAME, "anonymous")


def _resolve_client_ip(request: Request) -> str | None:
    """Resolve one client IP with proxy awareness when headers are present."""

    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()

    if request.client is None:
        return None
    return request.client.host


def _resolve_route_path(request: Request) -> str | None:
    """Return the FastAPI route template for the current request when available."""

    route = request.scope.get("route")
    return getattr(route, "path", None)


def _sanitize_value(value: Any) -> Any:
    """Reduce one value into a JSON-safe, bounded representation."""

    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 20:
                sanitized["__truncated__"] = True
                break
            normalized_key = str(key)
            sanitized[normalized_key] = (
                "[REDACTED]"
                if _is_sensitive_key(normalized_key)
                else _sanitize_value(item)
            )
        return sanitized

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized_items = [_sanitize_value(item) for item in items[:20]]
        if len(items) > 20:
            sanitized_items.append("__truncated__")
        return sanitized_items

    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, str):
            return _truncate_text(value)
        return value

    return _truncate_text(str(value))


def _is_sensitive_key(key: str) -> bool:
    """Identify credential-like payload keys that must not be persisted."""

    normalized = key.lower().replace("-", "_")
    return normalized in {
        "api_key",
        "authorization",
        "password",
        "token",
        "refresh_token",
    }


def _truncate_text(value: str) -> str:
    """Clamp one string to the configured logging length."""

    if len(value) <= settings.REQUEST_LOG_MAX_BODY_LENGTH:
        return value
    return f"{value[: settings.REQUEST_LOG_MAX_BODY_LENGTH]}..."
