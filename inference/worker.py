"""Indexing worker: Redis stream -> vLLM (caption + embedding) -> turbopuffer.

Every container runs one of these in the same Redis consumer group, so Redis
hands each scene to exactly one worker. Run as many containers as you have GPUs.

A job is acked only after its row is in turbopuffer. If a worker dies mid-job,
the job stays pending and another worker reclaims it after CLAIM_IDLE_MS.
Jobs that fail MAX_ATTEMPTS times move to the "<stream>:dead" stream.

Settings (env):
    REDIS_URL         redis://localhost:6379
    STREAM            scenes:index
    GROUP             indexers
    CONSUMER          $HOSTNAME             unique per container
    CONCURRENCY       8                     jobs in flight; vLLM batches them on the GPU
    MAX_ATTEMPTS      3
    CLAIM_IDLE_MS     120000                reclaim jobs pending this long
    VLM_URL           http://127.0.0.1:8000/v1
    VLM_MODEL         Qwen/Qwen3-VL-8B-Instruct-FP8
    EMBED_URL         http://127.0.0.1:8001/v1
    EMBED_MODEL       Qwen/Qwen3-Embedding-0.6B
    TURBOPUFFER_API_KEY, TURBOPUFFER_REGION (e.g. gcp-us-central1)
    TPUF_NAMESPACE    scenes
    HEALTH_PORT       8080
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import time
from urllib.parse import urlsplit

import redis.asyncio as redis
from aiohttp import ClientSession, ClientTimeout, web
from openai import AsyncOpenAI
from turbopuffer import AsyncTurbopuffer

from prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_context, search_text

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
STREAM = os.environ.get("STREAM", "scenes:index")
DEAD_STREAM = f"{STREAM}:dead"
GROUP = os.environ.get("GROUP", "indexers")
CONSUMER = os.environ.get("CONSUMER") or socket.gethostname()
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "3"))
CLAIM_IDLE_MS = int(os.environ.get("CLAIM_IDLE_MS", "120000"))
# Heartbeat + reclaim cadence; must be well under CLAIM_IDLE_MS so live jobs never look idle.
RECLAIM_EVERY_S = min(30.0, CLAIM_IDLE_MS / 4000)
VLM_URL = os.environ.get("VLM_URL", "http://127.0.0.1:8000/v1")
VLM_MODEL = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct-FP8")
EMBED_URL = os.environ.get("EMBED_URL", "http://127.0.0.1:8001/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
TPUF_NAMESPACE = os.environ.get("TPUF_NAMESPACE", "scenes")
HEALTH_PORT = int(os.environ.get("HEALTH_PORT", "8080"))

# Created once per namespace on first write; resending it unchanged is a no-op.
TPUF_SCHEMA = {
    "text": {"type": "string", "full_text_search": True},
    "labels": {"type": "[]string"},
    "objects_json": {"type": "string", "filterable": False},
}

log = logging.getLogger("worker")


class PermanentError(Exception):
    """The job can never succeed (malformed); don't retry it."""


class Worker:
    def __init__(self) -> None:
        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.vlm = AsyncOpenAI(base_url=VLM_URL, api_key="unused", timeout=120)
        self.embedder = AsyncOpenAI(base_url=EMBED_URL, api_key="unused", timeout=60)
        self.tpuf = AsyncTurbopuffer(region=os.environ.get("TURBOPUFFER_REGION"))
        self.ns = self.tpuf.namespace(TPUF_NAMESPACE)
        self.in_flight: set[asyncio.Task] = set()
        self.active_ids: set[str] = set()  # message ids this worker is processing right now
        self.last_tick = time.monotonic()
        self.done = self.failed = 0

    # ---------------------------------------------------------------- queue

    async def ensure_group(self) -> None:
        try:
            await self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
            log.info("created consumer group %r on %r", GROUP, STREAM)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def consume(self) -> None:
        while True:
            self.last_tick = time.monotonic()
            if len(self.in_flight) >= CONCURRENCY:
                await asyncio.wait(self.in_flight, return_when=asyncio.FIRST_COMPLETED)
                continue
            replies = await self.redis.xreadgroup(
                GROUP, CONSUMER, {STREAM: ">"}, count=CONCURRENCY - len(self.in_flight), block=5000
            )
            for _stream, messages in replies or []:
                for msg_id, fields in messages:
                    self.start(msg_id, fields)

    async def reclaim(self) -> None:
        """Keep our own jobs fresh, and take over jobs a crashed worker left pending."""
        while True:
            await asyncio.sleep(RECLAIM_EVERY_S)
            if self.active_ids:
                # Heartbeat: XCLAIM to ourselves with JUSTID resets the idle time without
                # counting a delivery, so slow jobs aren't stolen or charged extra attempts.
                await self.redis.xclaim(STREAM, GROUP, CONSUMER, min_idle_time=0,
                                        message_ids=list(self.active_ids), justid=True)
            start = "0-0"
            while True:
                # Redis 7 also returns deleted ids; those are already dropped from pending.
                start, messages, *_ = await self.redis.xautoclaim(
                    STREAM, GROUP, CONSUMER, min_idle_time=CLAIM_IDLE_MS, start_id=start, count=50
                )
                for msg_id, fields in messages:
                    if msg_id in self.active_ids:  # ours and still running, just slow
                        continue
                    if not fields:  # trimmed from the stream while pending
                        await self.redis.xack(STREAM, GROUP, msg_id)
                        continue
                    pending = await self.redis.xpending_range(STREAM, GROUP, min=msg_id, max=msg_id, count=1)
                    attempts = pending[0]["times_delivered"] if pending else 1
                    if attempts > MAX_ATTEMPTS:
                        await self.dead_letter(msg_id, fields, f"gave up after {attempts - 1} attempts")
                    else:
                        log.info("reclaimed %s (attempt %d)", msg_id, attempts)
                        self.start(msg_id, fields)
                if start == "0-0":
                    break

    def start(self, msg_id: str, fields: dict) -> None:
        task = asyncio.create_task(self.handle(msg_id, fields))
        self.in_flight.add(task)
        self.active_ids.add(msg_id)

        def finished(t: asyncio.Task) -> None:
            self.in_flight.discard(t)
            self.active_ids.discard(msg_id)

        task.add_done_callback(finished)

    async def handle(self, msg_id: str, fields: dict) -> None:
        try:
            scene = self.parse(fields)
            await self.index(scene)
        except PermanentError as exc:
            await self.dead_letter(msg_id, fields, str(exc))
        except Exception:
            # Left pending on purpose: reclaim() retries it after CLAIM_IDLE_MS.
            self.failed += 1
            log.exception("job %s failed, will retry", msg_id)
        else:
            await self.redis.xack(STREAM, GROUP, msg_id)
            self.done += 1

    async def dead_letter(self, msg_id: str, fields: dict, reason: str) -> None:
        log.error("dead-lettering %s: %s", msg_id, reason)
        await self.redis.xadd(DEAD_STREAM, {**fields, "reason": reason, "source_id": msg_id},
                              maxlen=10_000, approximate=True)
        await self.redis.xack(STREAM, GROUP, msg_id)

    @staticmethod
    def parse(fields: dict) -> dict:
        try:
            scene = json.loads(fields["job"])
            for key in ("ts_ms", "identity", "image", "image_b64"):
                scene[key]  # noqa: B018 - presence check
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise PermanentError(f"malformed job: {exc!r}") from exc
        return scene

    # ---------------------------------------------------------------- inference

    async def index(self, scene: dict) -> None:
        # Scenes saved before "id" existed are named <ts_ms>_<identity>.
        doc_id = scene.get("id") or f"{scene['ts_ms']}_{scene['identity']}"
        t0 = time.monotonic()

        reply = await self.vlm.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{scene['image_b64']}"}},
                    {"type": "text", "text": build_context(scene)},
                ]},
            ],
            temperature=0,
            max_tokens=512,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "scene_description", "schema": RESPONSE_SCHEMA}},
        )
        description = json.loads(reply.choices[0].message.content)
        t1 = time.monotonic()

        text = search_text(description)
        embedding = await self.embedder.embeddings.create(model=EMBED_MODEL, input=[text])
        t2 = time.monotonic()

        row = {
            "id": doc_id,
            "vector": embedding.data[0].embedding,
            "text": text,
            "caption": description["caption"],
            "place": description["place"],
            # Detector labels plus anything the VLM named, for exact filters.
            "labels": sorted({o["label"] for o in scene.get("objects") or []}
                             | {o["label"] for o in description["objects"]}),
            "objects_json": json.dumps({"detected": scene.get("objects") or [], "described": description["objects"]}),
            "kind": scene.get("kind", "arrived"),  # "left" rows are the best answer to "where is X"
            "ts_ms": int(scene["ts_ms"]),
            "identity": scene["identity"],
            "image": scene["image"],
        }
        if loc := scene.get("location"):
            row.update(lat=float(loc["lat"]), lon=float(loc["lon"]), location_accuracy_m=float(loc["accuracy_m"]))

        await self.ns.write(upsert_rows=[row], distance_metric="cosine_distance", schema=TPUF_SCHEMA)
        t3 = time.monotonic()
        log.info("indexed %s  vlm %.2fs embed %.2fs tpuf %.2fs  %s",
                 doc_id, t1 - t0, t2 - t1, t3 - t2, description["caption"])

    # ---------------------------------------------------------------- health

    async def serve_health(self) -> None:
        session = ClientSession(timeout=ClientTimeout(total=3))

        async def healthz(_request: web.Request) -> web.Response:
            problems = []
            if time.monotonic() - self.last_tick > 60:
                problems.append("queue loop stalled")
            for name, url in (("vlm", VLM_URL), ("embed", EMBED_URL)):
                try:
                    async with session.get(url.removesuffix("/v1") + "/health") as resp:
                        if resp.status != 200:
                            problems.append(f"{name} unhealthy ({resp.status})")
                except Exception as exc:
                    problems.append(f"{name} unreachable ({exc.__class__.__name__})")
            body = {"ok": not problems, "problems": problems, "consumer": CONSUMER,
                    "in_flight": len(self.in_flight), "done": self.done, "failed": self.failed}
            return web.json_response(body, status=503 if problems else 200)

        app = web.Application()
        app.router.add_get("/healthz", healthz)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", HEALTH_PORT).start()

    async def run(self) -> None:
        await self.ensure_group()
        await self.serve_health()
        log.info("consumer %r on redis %s stream %r, concurrency %d",
                 CONSUMER, urlsplit(REDIS_URL).hostname, STREAM, CONCURRENCY)
        # If either loop crashes, exit so the container restarts.
        await asyncio.gather(self.consume(), self.reclaim())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    asyncio.run(Worker().run())
