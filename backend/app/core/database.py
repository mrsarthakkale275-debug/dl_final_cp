"""Database session management for SQLAlchemy."""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.core.config import settings


def _connect_args(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    if "postgresql" in url or "postgres" in url:
        # Supabase/PostgreSQL require TLS. prepare_threshold=None keeps SQLAlchemy
        # compatible with transaction poolers such as Supabase PgBouncer.
        return {"sslmode": "require", "prepare_threshold": None}
    return {}


def _make_engine(url: str):
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=300,
        future=True,
        connect_args=_connect_args(url),
    )


engine = _make_engine(settings.database_url)

# Schema initialization uses the direct/session-pooler URL when one is configured.
direct_engine = (
    _make_engine(settings.direct_url)
    if settings.direct_url and settings.direct_url != settings.database_url
    else engine
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_database() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def init_db() -> None:
    from backend.app.models import inspection as _models  # noqa: F401

    target = direct_engine
    Base.metadata.create_all(bind=target)
    # Keep the explicit creation for environments where module reloading separates
    # model classes from the active declarative base.
    for table in (_models.Inspection.__table__, _models.InspectionResult.__table__):
        try:
            table.create(bind=target, checkfirst=True)
        except Exception:
            pass
