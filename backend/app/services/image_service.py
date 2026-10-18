"""Image upload validation and controlled storage."""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, UnidentifiedImageError

from backend.app.core.config import settings
from backend.app.core.logging_config import get_logger
from backend.app.utils.security import (
    ensure_relative_no_traversal,
    safe_extension,
    sanitize_filename,
    sha256_bytes,
)

logger = get_logger(__name__)


class ImageValidationError(Exception):
    """Raised when an uploaded image fails validation."""


@dataclass
class ValidatedImage:
    original_filename: str
    stored_relative_path: Optional[str]
    size_bytes: int
    content_type: Optional[str]
    sha256: str
    width: int
    height: int
    image_format: str
    image: Image.Image  # kept in memory briefly; caller should copy if needed


def _validate_content_type(content_type: Optional[str]) -> None:
    if content_type and content_type not in settings.allowed_image_types:
        raise ImageValidationError(
            f"Unsupported content type '{content_type}'. "
            f"Allowed: {sorted(settings.allowed_image_types)}"
        )


def validate_upload(file_bytes: bytes, filename: Optional[str], content_type: Optional[str]) -> ValidatedImage:
    """Validate an uploaded file and return a PIL image plus metadata."""

    if not file_bytes:
        raise ImageValidationError("Empty file upload is not allowed.")

    if len(file_bytes) > settings.max_upload_bytes:
        raise ImageValidationError(
            f"File size {len(file_bytes)} bytes exceeds limit of {settings.max_upload_bytes} bytes."
        )

    _validate_content_type(content_type)

    digest = sha256_bytes(file_bytes)

    # Try to open and fully decode the image. This rejects corrupt files.
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError("Uploaded file is not a valid or decodable image.") from exc

    fmt = image.format or "UNKNOWN"
    if fmt.upper() not in {"JPEG", "PNG", "BMP", "TIFF", "WEBP"}:
        raise ImageValidationError(f"Unsupported image format: {fmt}")

    if image.width <= 0 or image.height <= 0:
        raise ImageValidationError("Image has invalid dimensions.")

    # Normalize content type based on detected format if missing.
    if not content_type:
        content_type = f"image/{fmt.lower()}"
        if fmt.upper() == "JPEG":
            content_type = "image/jpeg"

    return ValidatedImage(
        original_filename=sanitize_filename(filename),
        stored_relative_path=None,
        size_bytes=len(file_bytes),
        content_type=content_type,
        sha256=digest,
        width=image.width,
        height=image.height,
        image_format=fmt,
        image=image,
    )


def store_image(validated: ValidatedImage, file_bytes: bytes) -> Optional[str]:
    """Store an image if not already present. Returns relative path or None on failure.

    Deduplication is by SHA-256 content hash. If an identical file already exists at
    the deterministic storage location, no overwrite occurs.
    """

    ext = safe_extension(validated.original_filename, validated.image_format)
    relative = validated.sha256[:2] + "/" + validated.sha256 + ext
    base = settings.image_storage_dir
    target = ensure_relative_no_traversal(base, relative)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        # Assume same hash means same content; do not rewrite.
        return relative.replace("\\", "/")

    try:
        target.write_bytes(file_bytes)
        logger.info("Stored inspection image sha256=%s path=%s", validated.sha256, relative)
        return relative.replace("\\", "/")
    except OSError as exc:
        logger.error("Failed to store image: %s", exc)
        return None


def read_stored_image(relative_path: str) -> Optional[bytes]:
    try:
        base = settings.image_storage_dir
        target = ensure_relative_no_traversal(base, relative_path)
        if not target.exists() or not target.is_file():
            return None
        return target.read_bytes()
    except (ValueError, OSError):
        return None


# Keep a tiny timing helper for services.
def now_ms() -> float:
    return time.perf_counter() * 1000.0
