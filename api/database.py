"""Database helpers for relational persistence."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from api.config import settings


class Base(DeclarativeBase):
    """Declarative base for SQLAlchemy models."""


engine = create_engine(
    settings.DATABASE_URL,
    future=True,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def init_database() -> None:
    """Create relational tables required by the API."""

    from api.services.request_log_service import RequestLog

    Base.metadata.create_all(bind=engine, tables=[RequestLog.__table__])


@contextmanager
def session_scope() -> Iterator[Session]:
    """Provide one transactional SQLAlchemy session."""

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
