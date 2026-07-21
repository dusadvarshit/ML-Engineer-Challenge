"""Shared, bounded validation for uploaded inference images."""

from __future__ import annotations

import warnings
from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from api.config import settings


async def read_validated_image(file: UploadFile) -> bytes:
    """Read a supported image once and reject unsafe or malformed payloads."""

    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415,
            detail={
                "code": "unsupported_media_type",
                "message": "Uploaded file must be a JPEG, PNG, or WebP image.",
            },
        )

    payload = await file.read(settings.MAX_FILE_SIZE + 1)
    if not payload:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_image", "message": "Uploaded image is empty."},
        )
    if len(payload) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "image_too_large",
                "message": f"Uploaded image exceeds the {settings.MAX_FILE_SIZE} byte limit.",
            },
        )

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                width, height = image.size
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unsafe_image",
                "message": "Uploaded image exceeds the decompression safety limit.",
            },
        ) from None
    except (OSError, SyntaxError, UnidentifiedImageError):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_image",
                "message": "Uploaded image could not be decoded.",
            },
        ) from None
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if not width or not height or max(width, height) > settings.MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_image_dimensions",
                "message": (
                    "Uploaded image dimensions must be positive and no larger than "
                    f"{settings.MAX_IMAGE_DIMENSION} pixels on either side."
                ),
            },
        )
    return payload
