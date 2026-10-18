"""Security helpers for uploads and paths."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Optional

# Characters allowed in filenames. Everything else is replaced.
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")
_MAX_NAME_LEN = 80


def sanitize_filename(name: Optional[str]) -> str:
    if not name:
        return "upload"
    name = Path(name).name  # drop any path components
    name = _SAFE_NAME_RE.sub("_", name)
    name = name.strip("._") or "upload"
    if len(name) > _MAX_NAME_LEN:
        stem, suffix = Path(name).stem, Path(name).suffix
        name = stem[: _MAX_NAME_LEN - len(suffix)] + suffix
    return name


def safe_extension(filename: Optional[str], detected_format: Optional[str]) -> str:
    ext = Path(filename or "").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if ext in allowed:
        return ext
    fmt = (detected_format or "").lower()
    mapping = {
        "jpeg": ".jpg",
        "png": ".png",
        "bmp": ".bmp",
        "tiff": ".tiff",
        "webp": ".webp",
    }
    return mapping.get(fmt, ".bin")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_storage_name(ext: str, content_hash: str) -> str:
    return f"{content_hash[:2]}/{content_hash}{ext}"


def ensure_relative_no_traversal(base: Path, relative: str) -> Path:
    resolved = (base / relative).resolve()
    base_resolved = base.resolve()
    if base_resolved not in resolved.parents and resolved != base_resolved:
        raise ValueError("Path traversal detected")
    return resolved
