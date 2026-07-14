"""Celery worker configuration for background jobs."""

from __future__ import annotations

from celery import Celery

from api.config import settings
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


@celery_app.task(name='api.persist_request_log')
def persist_request_log(payload: dict[str, object]) -> None:
    """Store one HTTP request log entry in Postgres."""

    save_request_log(payload)
