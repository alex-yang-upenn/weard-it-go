"""Consume Redis scene jobs and embed object crops with local Qwen3-VL-Embedding.

Run from the repository root after Redis and the local Qwen model are available:
    REDIS_URL=redis://localhost:6379 python backend/embedding_worker.py

Scene jobs use the same Redis stream and payload as inference/worker.py, but
this worker uses its own consumer group so both workers receive every scene.
It crops each object's normalized [x, y, width, height] box from the queued
image, then appends one JSON record per embedding to
backend/object_embeddings.jsonl. Use --once to drain available jobs.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import redis.asyncio as redis

BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = Path(os.environ.get("EMBEDDINGS_PATH", BACKEND_DIR / "object_embeddings.jsonl"))
MODEL_ID = os.environ.get("QWEN_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B")
MODEL_PATH = Path(os.environ.get("QWEN_EMBEDDING_MODEL_PATH", BACKEND_DIR / "models" / "Qwen3-VL-Embedding-2B"))
QWEN_REPO = Path(os.environ.get("QWEN_EMBEDDING_REPO", BACKEND_DIR / "Qwen3-VL-Embedding"))
MODEL = "qwen3-vl-embedding-2b"
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
STREAM = os.environ.get("STREAM", "scenes:index")
GROUP = os.environ.get("QWEN_EMBEDDING_GROUP", "qwen-embedders")
CONSUMER = os.environ.get("QWEN_EMBEDDING_CONSUMER") or f"{socket.gethostname()}-{os.getpid()}"
CLAIM_IDLE_MS = int(os.environ.get("QWEN_CLAIM_IDLE_MS", "120000"))
MAX_ATTEMPTS = int(os.environ.get("QWEN_MAX_ATTEMPTS", "3"))
RECLAIM_EVERY_S = min(30.0, CLAIM_IDLE_MS / 4000)
EMBEDDING_INSTRUCTION = os.environ.get(
    "QWEN_EMBEDDING_INSTRUCTION",
    "Represent this object crop for visual instance retrieval.",
)
PADDING_FRACTION = float(os.environ.get("EMBEDDING_CROP_PADDING", "0.12"))

log = logging.getLogger("embedding_worker")


@dataclass(frozen=True)
class CropJob:
    key: str
    redis_message_id: str
    scene_id: str
    image_name: str
    ts_ms: int | None
    identity: str | None
    object_index: int
    label: str
    confidence: float | None
    box: list[float]
    crop_bytes: bytes


def _crop_bytes(image: Image.Image, box: list[float]) -> bytes:
    """Crop normalized xywh coordinates from a decoded scene image."""
    if len(box) != 4:
        raise ValueError(f"expected xywh box with 4 values, got {box!r}")
    image = image.convert("RGB")
    width, height = image.size
    x, y, w, h = (float(value) for value in box)
    if w <= 0 or h <= 0:
        raise ValueError(f"box has non-positive dimensions: {box!r}")
    left, top = max(0.0, x) * width, max(0.0, y) * height
    right, bottom = min(1.0, x + w) * width, min(1.0, y + h) * height
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


async def embed_jobs(embedder: LocalEmbedder, jobs: list[CropJob]) -> bool:
    """Generate each vector locally and append it with the scene metadata."""
    import asyncio
    succeeded = True
    for job in jobs:
        try:
            vector = await asyncio.to_thread(embedder.embed, job.crop_bytes)
            append_record({
                "key": job.key,
                "status": "embedded",
                "model": MODEL_ID,
                "dimensions": len(vector),
                "embedding": vector,
                "redis_message_id": job.redis_message_id,
                "scene": job.scene_id,
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
            succeeded = False
            log.exception("embedding failed for %s", job.key)
            append_record({"key": job.key, "status": "error", "error": str(exc)})
    return succeeded


def parse_scene(fields: dict[str, str]) -> dict[str, Any]:
    try:
        scene = json.loads(fields["job"])
        for required in ("ts_ms", "identity", "image", "image_b64"):
            scene[required]
        if not isinstance(scene.get("objects"), list):
            raise ValueError("scene objects must be a list")
        return scene
    except (KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"malformed Redis scene job: {exc}") from exc


def scene_jobs(scene: dict[str, Any], message_id: str, completed: set[str]) -> list[CropJob]:
    try:
        image_data = base64.b64decode(scene["image_b64"], validate=True)
        with Image.open(BytesIO(image_data)) as decoded:
            image = decoded.convert("RGB")
    except Exception as exc:
        raise ValueError(f"could not decode queued scene image: {exc}") from exc

    scene_id = str(scene.get("id") or f"{scene['ts_ms']}_{scene['identity']}")
    jobs: list[CropJob] = []
    for index, obj in enumerate(scene.get("objects") or []):
        key = f"{MODEL}:{scene_id}:{index}"
        if key in completed:
            continue
        box = obj.get("box")
        if not isinstance(box, list):
            log.warning("skipping object without box in scene %s object %d", scene_id, index)
            continue
        crop = _crop_bytes(image, box)
        jobs.append(CropJob(
            key=key,
            redis_message_id=message_id,
            scene_id=scene_id,
            image_name=str(scene["image"]),
            ts_ms=scene.get("ts_ms"),
            identity=scene.get("identity"),
            object_index=index,
            label=str(obj.get("label", "unknown")),
            confidence=obj.get("confidence"),
            box=[float(v) for v in box],
            crop_bytes=crop,
        ))
    return jobs


class RedisEmbeddingWorker:
    def __init__(self, embedder: LocalEmbedder, once: bool) -> None:
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.embedder = embedder
        self.once = once
        self.last_reclaim = 0.0

    # Ensure the Redis consumer group exists, creating it if necessary.
    async def ensure_group(self) -> None:
        try:
            # A new group starts at the oldest retained stream entry for backfill.
            await self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
            log.info("created Redis consumer group %r on %r", GROUP, STREAM)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    # Process a single message from the Redis stream, handling success, failure, and dead-lettering.
    async def process(self, message_id: str, fields: dict[str, str]) -> None:
        try:
            scene = parse_scene(fields)
            jobs = scene_jobs(scene, message_id, load_completed())
            success = await embed_jobs(self.embedder, jobs)
            if success:
                await self.redis.xack(STREAM, GROUP, message_id)
            else:
                log.warning("leaving %s pending so failed crops can retry", message_id)
        except ValueError as exc:
            log.error("dead-lettering malformed scene %s: %s", message_id, exc)
            await self.redis.xadd(f"{STREAM}:dead", {
                **fields, "source_id": message_id, "reason": str(exc),
            }, maxlen=10_000, approximate=True)
            await self.redis.xack(STREAM, GROUP, message_id)
        except Exception:
            log.exception("scene job %s failed; leaving it pending for retry", message_id)

    # Process a message while sending periodic heartbeats to keep the claim alive.
    async def process_with_heartbeat(self, message_id: str, fields: dict[str, str]) -> None:
        import asyncio

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(RECLAIM_EVERY_S)
                await self.redis.xclaim(
                    STREAM, GROUP, CONSUMER, min_idle_time=0,
                    message_ids=[message_id], justid=True,
                )

        task = asyncio.create_task(heartbeat())
        try:
            await self.process(message_id, fields)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    # Reclaim messages that have been idle for too long, dead-lettering those that exceed the retry limit.
    async def reclaim(self) -> None:
        cursor = "0-0"
        while True:
            cursor, messages, *_ = await self.redis.xautoclaim(
                STREAM, GROUP, CONSUMER, min_idle_time=CLAIM_IDLE_MS,
                start_id=cursor, count=10,
            )
            for message_id, fields in messages:
                pending = await self.redis.xpending_range(
                    STREAM, GROUP, min=message_id, max=message_id, count=1,
                )
                attempts = pending[0]["times_delivered"] if pending else 1
                if attempts > MAX_ATTEMPTS:
                    log.error("dead-lettering %s after %d deliveries", message_id, attempts)
                    await self.redis.xadd(f"{STREAM}:dead", {
                        **fields, "source_id": message_id,
                        "reason": f"gave up after {attempts} deliveries",
                    }, maxlen=10_000, approximate=True)
                    await self.redis.xack(STREAM, GROUP, message_id)
                else:
                    log.info("retrying pending scene %s (delivery %d)", message_id, attempts)
                    await self.process_with_heartbeat(message_id, fields)
            if cursor == "0-0":
                break

    # Run the consumer loop, reclaiming stale messages and processing new ones.
    async def run(self) -> None:
        await self.ensure_group()
        log.info("consumer %r reading stream %r in group %r", CONSUMER, STREAM, GROUP)
        while True:
            now = time.monotonic()
            if now - self.last_reclaim >= RECLAIM_EVERY_S:
                await self.reclaim()
                self.last_reclaim = time.monotonic()
            replies = await self.redis.xreadgroup(
                GROUP, CONSUMER, {STREAM: ">"}, count=1, block=1000,
            )
            entries = [(message_id, fields) for _stream, messages in replies or []
                       for message_id, fields in messages]
            for message_id, fields in entries:
                await self.process_with_heartbeat(message_id, fields)
            if self.once and not entries:
                return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="drain available Redis jobs and exit")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    try:
        embedder = LocalEmbedder()
        asyncio.run(RedisEmbeddingWorker(embedder, args.once).run())
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
