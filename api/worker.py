"""Celery worker configuration for background jobs."""

from __future__ import annotations

import base64
import logging
from time import perf_counter
from typing import NoReturn, Protocol

from celery import Celery  # type: ignore[import-untyped]

from api.config import settings
from api.metrics import observe_batch_request, observe_inference
from api.models.object_detection import ObjectDetectionModel
from api.services.batch_job_service import (
    complete_batch_job,
    fail_batch_job,
    mark_batch_job_running,
)
from api.services.object_detection.registry import get_object_detection_service
from api.services.request_log_service import save_request_log

logger = logging.getLogger(__name__)


class RetryableTask(Protocol):
    """Minimal Celery task contract used by the bound batch task."""

    request: object

    def retry(self, *, exc: Exception, countdown: int) -> NoReturn:
        """Request a retry and raise the task retry signal."""


celery_app = Celery(
    "ml_api",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery_app.conf.update(
    accept_content=["json"],
    result_serializer="json",
    task_ignore_result=True,
    task_serializer="json",
    task_time_limit=settings.BATCH_TASK_TIME_LIMIT_SECONDS,
)


@celery_app.task(name="api.persist_request_log")
def persist_request_log(payload: dict[str, object]) -> None:
    """Store one HTTP request log entry in Postgres."""

    save_request_log(payload)


@celery_app.task(
    bind=True,
    name="api.run_batch_detection",
    max_retries=settings.BATCH_TASK_MAX_RETRIES,
)
def run_batch_detection(
    self: RetryableTask,
    job_id: str,
    encoded_payloads: list[str],
) -> None:
    """Run one durable object-detection batch and persist its result.

    Image bytes travel only in the broker message; the relational job record
    intentionally stores metadata and results, never original uploads.
    """

    job = mark_batch_job_running(job_id)
    if job is None:
        return

    started_at = perf_counter()
    try:
        payloads = [
            base64.b64decode(payload, validate=True)
            for payload in encoded_payloads
        ]
        model = ObjectDetectionModel(job.model)
        service = get_object_detection_service(model)
        model_version = service.get_model_version()
        detections_by_image = service.predict_batch_from_bytes(payloads)
        if len(detections_by_image) != len(payloads):
            raise RuntimeError(
                "Model returned a result count different from the submitted batch."
            )
        complete_batch_job(
            job_id=job_id,
            model_version=model_version,
            items=[
                {"detections": [item.model_dump() for item in detections]}
                for detections in detections_by_image
            ],
        )
        observe_inference(
            task=job.task,
            model=model_version,
            outcome="success",
            duration_seconds=perf_counter() - started_at,
            image_count=len(payloads),
        )
        observe_batch_request(
            task=job.task, batch_size=len(payloads), outcome="success"
        )
    except Exception as exc:
        observe_batch_request(
            task=job.task, batch_size=len(encoded_payloads), outcome="error"
        )
        retries = getattr(getattr(self, "request", None), "retries", 0)
        if retries < settings.BATCH_TASK_MAX_RETRIES:
            logger.warning(
                "Retrying batch job %s after worker error: %s", job_id, exc
            )
            raise self.retry(exc=exc, countdown=min(2 ** (retries + 1), 30))
        logger.exception("Batch job %s failed permanently", job_id)
        fail_batch_job(
            job_id=job_id,
            error="Batch inference failed. Please retry with a new request.",
        )
