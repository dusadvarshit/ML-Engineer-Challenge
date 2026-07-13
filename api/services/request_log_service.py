"""Persistence for API request logs."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base, session_scope


class RequestLog(Base):
    """Relational record for one HTTP request/response exchange."""

    __tablename__ = 'request_logs'

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    method: Mapped[str] = mapped_column(String(16), nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)
    route_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, default='anonymous')
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    response_content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    request_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    response_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


def save_request_log(payload: dict[str, Any]) -> None:
    """Persist one request log entry."""

    with session_scope() as session:
        session.add(
            RequestLog(
                request_id=str(payload['request_id']),
                method=str(payload['method']),
                path=str(payload['path']),
                route_path=payload.get('route_path'),
                status_code=int(payload['status_code']),
                latency_ms=float(payload['latency_ms']),
                user_id=str(payload.get('user_id') or 'anonymous'),
                client_ip=payload.get('client_ip'),
                user_agent=payload.get('user_agent'),
                request_content_type=payload.get('request_content_type'),
                response_content_type=payload.get('response_content_type'),
                request_payload=payload.get('request_payload'),
                response_payload=payload.get('response_payload'),
                error_message=payload.get('error_message'),
            )
        )
