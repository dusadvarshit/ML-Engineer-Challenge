"""Create request_logs table."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = '20260713_000001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the request log table and its supporting index."""

    op.create_table(
        'request_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('request_id', sa.String(length=64), nullable=False),
        sa.Column('method', sa.String(length=16), nullable=False),
        sa.Column('path', sa.String(length=512), nullable=False),
        sa.Column('route_path', sa.String(length=512), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=False),
        sa.Column('client_ip', sa.String(length=64), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('request_content_type', sa.String(length=255), nullable=True),
        sa.Column('response_content_type', sa.String(length=255), nullable=True),
        sa.Column('request_payload', sa.JSON(), nullable=True),
        sa.Column('response_payload', sa.JSON(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_request_logs_request_id', 'request_logs', ['request_id'], unique=False)


def downgrade() -> None:
    """Drop the request log table and index."""

    op.drop_index('ix_request_logs_request_id', table_name='request_logs')
    op.drop_table('request_logs')
