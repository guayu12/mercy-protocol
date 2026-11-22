import base64
import io
import os
from pathlib import Path

from PIL import Image, ImageOps


MAX_IMAGE_PIXELS = 50_000_000


class ImageTooLarge(ValueError):
    pass


def cache_limit_bytes() -> int:
    try:
        return max(0, int(os.environ.get("MEDIA_CACHE_MAX_MB", "500"))) * 1024 * 1024
    except ValueError:
        return 500 * 1024 * 1024


def image_data_url_from_path(path: str) -> str | None:
    image_path = Path(path)
    if not image_path.is_file():
        return None

    with Image.open(image_path) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ImageTooLarge("Image exceeds pixel limit")
        source.seek(0)
        image = ImageOps.exif_transpose(source.copy())
        has_alpha = "A" in image.getbands() or "transparency" in source.info
        if has_alpha:
            image = image.convert("RGBA")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        image.info.clear()
        image.getexif().clear()
        output = io.BytesIO()
        if has_alpha:
            image.save(output, format="PNG")
            mime = "image/png"
        else:
            image = image.convert("RGB")
            image.save(output, format="JPEG", quality=85)
            mime = "image/jpeg"

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
