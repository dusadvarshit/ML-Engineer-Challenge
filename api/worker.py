"""Celery worker configuration for background jobs."""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import worker_process_init

from api.config import settings
from api.database import init_database
from api.services.request_log_service import save_request_log

celery_app = Celery(
    'ml_api',
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    accept_content=['json'],
    result_serializer='json',
    task_ignore_result=True,
    task_serializer='json',
)

_database_initialized = False


def ensure_database_initialized() -> None:
    """Create request log tables once per worker process."""

    global _database_initialized
    if _database_initialized:
        return

    init_database()
    _database_initialized = True


@worker_process_init.connect
def _initialize_worker_dependencies(**_: Any) -> None:
    """Prepare worker dependencies after a child process starts."""

    ensure_database_initialized()


@celery_app.task(name='api.persist_request_log')
def persist_request_log(payload: dict[str, Any]) -> None:
    """Store one HTTP request log entry in Postgres."""

    ensure_database_initialized()
    save_request_log(payload)
