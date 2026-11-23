import io

import numpy as np
import zstandard
from PIL import Image


def compress_raw_frame(raw_frame_bytes: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=1).compress(raw_frame_bytes)


def decompress_raw_frame(compressed_frame_bytes: bytes) -> bytes:
    return zstandard.ZstdDecompressor().decompress(compressed_frame_bytes)


def upsample_chroma_plane(chroma_plane: np.ndarray, width: int, height: int) -> np.ndarray:
    return chroma_plane.repeat(2, axis=0).repeat(2, axis=1)[:height, :width].astype(np.float32) - 128


def convert_i420_to_rgb(i420_bytes: bytes, width: int, height: int) -> np.ndarray:
    chroma_width, chroma_height = (width + 1) // 2, (height + 1) // 2
    luma_size, chroma_size = width * height, chroma_width * chroma_height
    planes = np.frombuffer(i420_bytes, dtype=np.uint8)
    luma = planes[:luma_size].reshape(height, width).astype(np.float32)
    blue_difference = upsample_chroma_plane(
        planes[luma_size : luma_size + chroma_size].reshape(chroma_height, chroma_width), width, height
    )
    red_difference = upsample_chroma_plane(
        planes[luma_size + chroma_size : luma_size + 2 * chroma_size].reshape(chroma_height, chroma_width), width, height
    )
    red = luma + 1.402 * red_difference
    green = luma - 0.344136 * blue_difference - 0.714136 * red_difference
    blue = luma + 1.772 * blue_difference
    return np.clip(np.stack([red, green, blue], axis=-1), 0, 255).astype(np.uint8)


def encode_rgb_as_jpeg(rgb: np.ndarray, max_side_pixels: int | None = None) -> bytes:
    image = Image.fromarray(rgb)
    if max_side_pixels is not None:
        image.thumbnail((max_side_pixels, max_side_pixels))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()
