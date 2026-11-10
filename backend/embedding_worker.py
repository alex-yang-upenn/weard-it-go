"""Embed object crops from saved scenes with local Qwen3-VL-Embedding.

Run from the repository root or backend directory:
    git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git backend/Qwen3-VL-Embedding
    # Download Qwen/Qwen3-VL-Embedding-2B into backend/models/Qwen3-VL-Embedding-2B
    python backend/embedding_worker.py

The worker polls backend/scenes for scene JSON files, crops each object's
normalized [x, y, width, height] box, and appends one JSON record per embedding
to backend/object_embeddings.jsonl. It keeps running by default; use --once to
process the current backlog and exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parent
SCENES_DIR = Path(os.environ.get("SCENES_DIR", BACKEND_DIR / "scenes"))
OUTPUT_PATH = Path(os.environ.get("EMBEDDINGS_PATH", BACKEND_DIR / "object_embeddings.jsonl"))
MODEL_ID = os.environ.get("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
MODEL_PATH = Path(os.environ.get("QWEN_EMBEDDING_MODEL_PATH", BACKEND_DIR / "models" / "Qwen3-VL-Embedding-2B"))
QWEN_REPO = Path(os.environ.get("QWEN_EMBEDDING_REPO", BACKEND_DIR / "Qwen3-VL-Embedding"))
MODEL = "qwen3-vl-embedding-2b"
EMBEDDING_INSTRUCTION = os.environ.get(
    "QWEN_EMBEDDING_INSTRUCTION",
    "Represent this object crop for visual instance retrieval.",
)
POLL_SECONDS = float(os.environ.get("EMBEDDING_POLL_SECONDS", "3"))
PADDING_FRACTION = float(os.environ.get("EMBEDDING_CROP_PADDING", "0.12"))

log = logging.getLogger("embedding_worker")


@dataclass(frozen=True)
class CropJob:
    key: str
    scene_path: Path
    image_name: str
    ts_ms: int | None
    identity: str | None
    object_index: int
    label: str
    confidence: float | None
    box: list[float]
    crop_bytes: bytes


def _crop_bytes(image_path: Path, box: list[float]) -> bytes:
    """Crop normalized xywh coordinates and return JPEG bytes."""
    if len(box) != 4:
        raise ValueError(f"expected xywh box with 4 values, got {box!r}")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    x, y, w, h = (float(value) for value in box)
    if w <= 0 or h <= 0:
        raise ValueError(f"box has non-positive dimensions: {box!r}")
    left, top = x * width, y * height
    right, bottom = (x + w) * width, (y + h) * height
    pad_x, pad_y = (right - left) * PADDING_FRACTION, (bottom - top) * PADDING_FRACTION
    bounds = (
        max(0, int(left - pad_x)), max(0, int(top - pad_y)),
        min(width, int(right + pad_x)), min(height, int(bottom + pad_y)),
    )
    crop = image.crop(bounds)
    if crop.width == 0 or crop.height == 0:
        raise ValueError(f"box produced an empty crop: {box!r}")
    buffer = BytesIO()
    crop.save(buffer, format="JPEG", quality=92, optimize=True)
    return buffer.getvalue()


def load_jobs(completed: set[str]) -> list[CropJob]:
    jobs: list[CropJob] = []
    for scene_path in sorted(SCENES_DIR.glob("*.json")):
        try:
            scene = json.loads(scene_path.read_text())
            image_name = scene.get("image")
            image_path = (SCENES_DIR / image_name).resolve()
            if image_path.parent != SCENES_DIR.resolve() or not image_path.is_file():
                log.warning("scene image missing or outside scenes directory: %s", scene_path)
                continue
            for index, obj in enumerate(scene.get("objects") or []):
                key = f"{MODEL}:{scene_path.name}:{index}"
                if key in completed:
                    continue
                box = obj.get("box")
                if not isinstance(box, list):
                    log.warning("skipping object without box: %s object %d", scene_path.name, index)
                    continue
                try:
                    crop = _crop_bytes(image_path, box)
                except Exception as exc:
                    log.warning("crop failed for %s object %d: %s", scene_path.name, index, exc)
                    continue
                jobs.append(CropJob(
                    key=key,
                    scene_path=scene_path,
                    image_name=image_name,
                    ts_ms=scene.get("ts_ms"),
                    identity=scene.get("identity"),
                    object_index=index,
                    label=str(obj.get("label", "unknown")),
                    confidence=obj.get("confidence"),
                    box=[float(v) for v in box],
                    crop_bytes=crop,
                ))
        except Exception:
            log.exception("could not read scene %s", scene_path)
    return jobs


def load_completed() -> set[str]:
    completed: set[str] = set()
    if not OUTPUT_PATH.exists():
        return completed
    with OUTPUT_PATH.open() as records:
        for line in records:
            try:
                record = json.loads(line)
                if record.get("status") == "embedded":
                    completed.add(record["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def append_record(record: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a") as output:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


class LocalEmbedder:
    """Thin wrapper around Qwen's official embedding implementation."""

    def __init__(self) -> None:
        repo_path = QWEN_REPO.resolve()
        if not (repo_path / "src" / "models" / "qwen3_vl_embedding.py").is_file():
            raise SystemExit(
                f"Qwen embedding source not found at {repo_path}. Clone "
                "https://github.com/QwenLM/Qwen3-VL-Embedding there or set QWEN_EMBEDDING_REPO."
            )
        if not MODEL_PATH.exists():
            raise SystemExit(
                f"Model checkpoint not found at {MODEL_PATH}. Download Qwen/Qwen3-VL-Embedding-2B "
                "there or set QWEN_EMBEDDING_MODEL_PATH."
            )

        sys.path.insert(0, str(repo_path))
        try:
            import torch
            from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
        except ImportError as exc:
            raise SystemExit(
                "Local Qwen dependencies are missing. Install torch, transformers>=4.57.3, "
                "accelerate, and qwen-vl-utils>=0.0.14."
            ) from exc

        if torch.cuda.is_available():
            self.device = "cuda"
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
            dtype = torch.float16
        else:
            self.device = "cpu"
            dtype = torch.float32
        self.torch = torch
        self.model = Qwen3VLEmbedder(
            model_name_or_path=str(MODEL_PATH),
            torch_dtype=dtype,
        )
        # The official helper currently selects CUDA or CPU internally. Move to
        # MPS explicitly when available on Apple Silicon.
        if self.device == "mps":
            self.model.model.to(self.device)
        log.info("loaded %s on %s", MODEL_ID, self.device)

    def embed(self, crop_bytes: bytes) -> list[float]:
        with Image.open(BytesIO(crop_bytes)) as image:
            crop = image.convert("RGB")
        vectors = self.model.process([{
            "image": crop,
            "instruction": EMBEDDING_INSTRUCTION,
        }])
        return vectors[0].detach().float().cpu().tolist()


def embed_jobs(embedder: LocalEmbedder, jobs: list[CropJob]) -> None:
    """Generate each vector locally and append it with the scene metadata."""
    for job in jobs:
        try:
            vector = embedder.embed(job.crop_bytes)
            append_record({
                "key": job.key,
                "status": "embedded",
                "model": MODEL_ID,
                "dimensions": len(vector),
                "embedding": vector,
                "scene": job.scene_path.name,
                "image": job.image_name,
                "ts_ms": job.ts_ms,
                "identity": job.identity,
                "object_index": job.object_index,
                "label": job.label,
                "confidence": job.confidence,
                "box": job.box,
                "device": embedder.device,
            })
            log.info("embedded %s [%s], vector dimensions=%d", job.key, job.label, len(vector))
        except Exception as exc:
            log.exception("embedding failed for %s", job.key)
            append_record({"key": job.key, "status": "error", "error": str(exc)})


def run(once: bool) -> None:
    embedder = LocalEmbedder()
    while True:
        completed = load_completed()
        jobs = load_jobs(completed)
        if jobs:
            log.info("found %d object crops to embed", len(jobs))
            embed_jobs(embedder, jobs)
        if once:
            return
        import time
        time.sleep(POLL_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="process current scenes and exit")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    if not SCENES_DIR.is_dir():
        raise SystemExit(f"Scenes directory not found: {SCENES_DIR}")
    try:
        run(args.once)
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
