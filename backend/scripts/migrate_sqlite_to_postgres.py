#!/usr/bin/env python3
"""Explicit SQLite -> PostgreSQL migration helper for AdaptiveInspect AI.

Usage:
  python backend/scripts/migrate_sqlite_to_postgres.py --dry-run
  python backend/scripts/migrate_sqlite_to_postgres.py --source-path backend/adaptive_inspect.db
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, select, text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings
from backend.app.models.inspection import Inspection, InspectionResult
from backend.app.core.database import Base


def _sqlite_url(path: str) -> str:
    return f"sqlite:///{Path(path).resolve()}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-path", default="backend/adaptive_inspect.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    if not source_path.exists():
        raise SystemExit(f"SQLite source not found: {source_path}")

    source_engine = create_engine(_sqlite_url(str(source_path)), future=True)
    target_url = settings.direct_url if settings.direct_url != settings.database_url else os.getenv("DIRECT_URL", settings.database_url)
    if not target_url or target_url.startswith("sqlite"):
        raise SystemExit("A PostgreSQL target URL is required (DIRECT_URL or DATABASE_URL).")
    target_engine = create_engine(target_url, pool_pre_ping=True, future=True)

    Base.metadata.create_all(bind=target_engine)
    with source_engine.connect() as conn:
        source_inspections = conn.execute(select(Inspection)).scalars().all()
        source_results = conn.execute(select(InspectionResult)).scalars().all()

    print(f"Source inspections: {len(source_inspections)}")
    print(f"Source results: {len(source_results)}")

    if args.dry_run:
        print("Dry run only; no records were changed.")
        return

    with target_engine.begin() as conn:
        inserted_inspections = 0
        inserted_results = 0
        for row in source_inspections:
            data = {column: getattr(row, column) for column in Inspection.__table__.columns.keys()}
            result = conn.execute(
                Inspection.__table__.insert().values(**data).prefix_with("ON CONFLICT (id) DO NOTHING")
            )
            inserted_inspections += result.rowcount or 0
        for row in source_results:
            data = {column: getattr(row, column) for column in InspectionResult.__table__.columns.keys()}
            result = conn.execute(
                InspectionResult.__table__.insert().values(**data).prefix_with("ON CONFLICT (id) DO NOTHING")
            )
            inserted_results += result.rowcount or 0
        print(f"Target rows inserted: inspections={inserted_inspections}, results={inserted_results}")


if __name__ == "__main__":
    main()
