"""Durable batch-job persistence and worker state transitions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.config import settings
from api.database import Base, session_scope
from api.models.object_detection import (
    BatchJobStatus,
    InferenceTask,
    ObjectDetectionModel,
)


class BatchJob(Base):
    """Metadata for an asynchronous batch request; image bytes are never stored here."""

    __tablename__ = "batch_jobs"
    __table_args__ = (
        UniqueConstraint(
            "client_id",
            "idempotency_key",
            name="uq_batch_jobs_client_idempotency",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    client_id: Mapped[str] = mapped_column(
        String(255), nullable=False, index=True
    )
    idempotency_key: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    task: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    items: Mapped[list["BatchJobItemRecord"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="BatchJobItemRecord.position",
    )


class BatchJobItemRecord(Base):
    """Metadata/results for one submitted batch item."""

    __tablename__ = "batch_job_items"

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, autoincrement=True
    )
    job_id: Mapped[str] = mapped_column(
        ForeignKey("batch_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str | None] = mapped_column(
        String(255), nullable=True
    )
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    detections: Mapped[list[dict[str, Any]] | None] = mapped_column(
        JSON, nullable=True
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    job: Mapped[BatchJob] = relationship(back_populates="items")


def create_batch_job(
    *,
    client_id: str,
    task: InferenceTask,
    model: ObjectDetectionModel,
    item_metadata: list[dict[str, Any]],
    idempotency_key: str | None,
) -> tuple[BatchJob, bool]:
    """Create a queued job, or return the previous job for an idempotent replay."""

    now = datetime.now(timezone.utc)
    with session_scope() as session:
        if idempotency_key:
            existing = session.scalar(
                select(BatchJob).where(
                    BatchJob.client_id == client_id,
                    BatchJob.idempotency_key == idempotency_key,
                )
            )
            if existing is not None:
                _load_items(session, existing)
                return existing, True
        job = BatchJob(
            id=str(uuid4()),
            client_id=client_id,
            idempotency_key=idempotency_key,
            task=task.value,
            model=model.value,
            status=BatchJobStatus.QUEUED.value,
            attempts=0,
            created_at=now,
            expires_at=now
            + timedelta(seconds=settings.BATCH_RESULT_TTL_SECONDS),
        )
        for position, item in enumerate(item_metadata):
            job.items.append(BatchJobItemRecord(position=position, **item))
        session.add(job)
        session.flush()
        return job, False


def get_batch_job(*, job_id: str, client_id: str) -> BatchJob | None:
    """Return a job only to its submitting API client, expiring old results lazily."""

    with session_scope() as session:
        job = session.scalar(
            select(BatchJob).where(
                BatchJob.id == job_id, BatchJob.client_id == client_id
            )
        )
        if job is None:
            return None
        if job.status in {
            BatchJobStatus.SUCCEEDED.value,
            BatchJobStatus.FAILED.value,
        } and job.expires_at <= datetime.now(timezone.utc):
            job.status = BatchJobStatus.EXPIRED.value
        _load_items(session, job)
        return job


def mark_batch_job_running(job_id: str) -> BatchJob | None:
    """Claim a queued/retryable job for one worker attempt."""

    with session_scope() as session:
        job = session.get(BatchJob, job_id)
        if job is None or job.status in {
            BatchJobStatus.SUCCEEDED.value,
            BatchJobStatus.EXPIRED.value,
        }:
            return None
        job.status = BatchJobStatus.RUNNING.value
        job.attempts += 1
        job.started_at = datetime.now(timezone.utc)
        _load_items(session, job)
        return job


def complete_batch_job(
    *, job_id: str, model_version: str, items: list[dict[str, Any]]
) -> None:
    """Persist each result/failure and mark the completed batch successful."""

    with session_scope() as session:
        job = session.get(BatchJob, job_id)
        if job is None:
            return
        job.status = BatchJobStatus.SUCCEEDED.value
        job.model_version = model_version
        job.completed_at = datetime.now(timezone.utc)
        for record, item in zip(
            sorted(job.items, key=lambda value: value.position),
            items,
            strict=True,
        ):
            record.detections = item.get("detections")
            record.error = item.get("error")


def fail_batch_job(*, job_id: str, error: str) -> None:
    """Mark a job terminally failed without persisting a worker stack trace."""

    with session_scope() as session:
        job = session.get(BatchJob, job_id)
        if job is None:
            return
        job.status = BatchJobStatus.FAILED.value
        job.error = error[:2048]
        job.completed_at = datetime.now(timezone.utc)


def delete_batch_job(*, job_id: str) -> None:
    """Remove a job which could not be published to the task broker.

    A broker failure happens before any worker can see the job, so retaining it
    as queued would make polling clients wait forever.
    """

    with session_scope() as session:
        job = session.get(BatchJob, job_id)
        if job is not None:
            session.delete(job)


def _load_items(session: Any, job: BatchJob) -> None:
    """Load the relationship while its SQLAlchemy session remains open."""

    session.refresh(job, attribute_names=["items"])
