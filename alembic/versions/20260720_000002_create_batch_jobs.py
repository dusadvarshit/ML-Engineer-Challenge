"""Create durable batch job metadata tables."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "20260720_000002"
down_revision = "20260713_000001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create job and per-item result tables without storing image bytes."""

    op.create_table(
        "batch_jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("client_id", sa.String(length=255), nullable=False),
        sa.Column("idempotency_key", sa.String(length=255), nullable=True),
        sa.Column("task", sa.String(length=32), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=255), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("client_id", "idempotency_key", name="uq_batch_jobs_client_idempotency"),
    )
    op.create_index("ix_batch_jobs_client_id", "batch_jobs", ["client_id"], unique=False)
    op.create_table(
        "batch_job_items",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content_type", sa.String(length=255), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("detections", sa.JSON(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["batch_jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_batch_job_items_job_id", "batch_job_items", ["job_id"], unique=False)


def downgrade() -> None:
    """Drop batch job metadata tables."""

    op.drop_index("ix_batch_job_items_job_id", table_name="batch_job_items")
    op.drop_table("batch_job_items")
    op.drop_index("ix_batch_jobs_client_id", table_name="batch_jobs")
    op.drop_table("batch_jobs")
