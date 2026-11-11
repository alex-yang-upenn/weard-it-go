"""Live video agent for Wear'd It Go.

One process that:
  1. Serves access tokens to the iOS app:  GET /token?identity=<id>
     -> {"url": "ws://<this-host>:7880", "token": "...", "room": "glasses"}
     and takes the phone's GPS position:   POST /location
     {"identity": "...", "lat": ..., "lon": ..., "accuracy_m": ..., "ts_ms": ...}
     The phone's latest location is kept in memory and attached to queued scenes.
  2. Joins the same LiveKit room as "agent", subscribes to the glasses video,
     samples frames, runs YOLOE open-vocabulary detection on them, and pings
     the phone back over LiveKit RPC (method "alert") when objects come into view.
  3. Encodes selected "scenes" (frame + detections) in memory and queues them:
       arrived   objects just came into view (also sent to the phone as an alert)
       left      the last frame an object was in before it left view: where it was left
       snapshot  every SNAPSHOT_EVERY_S when the view has changed, whatever YOLOE saw,
                 so things YOLOE misses still get indexed

Run (with a local dev LiveKit server, see dev.sh):
    pip install -r requirements.txt
    python live_agent.py

Environment (defaults match `livekit-server --dev`):
    LIVEKIT_URL         ws://127.0.0.1:7880   where the agent itself connects
    LIVEKIT_API_KEY     devkey
    LIVEKIT_API_SECRET  secret
    LIVEKIT_PUBLIC_PORT 7880                  port handed to the phone
    ROOM_NAME           glasses
    TOKEN_PORT          8765
    ANALYZE_FPS         2                     frames per second fed to the detector
    YOLOE_MODEL         yoloe-26s-seg.pt      any YOLOE weights (n/s/m/l/x sizes)
    YOLOE_CLASSES       keys,wallet,...       comma-separated things to look for;
                                              "*" = prompt-free (name everything)
    YOLOE_CONF          0.35                  minimum detection confidence
    FORGET_AFTER_S      30                    re-announce an object after it's been
                                              out of view this long
    LEFT_VIEW_S         3                     unseen this long = it left view; save
                                              its last frame
    SNAPSHOT_EVERY_S    20                    at most one snapshot this often; 0 = off
    SNAPSHOT_MIN_CHANGE 0.08                  how different the view must be from the
                                              last queued scene (0-1, mean pixel change)
    REDIS_URL           required              e.g. redis://localhost:6379; selected scenes
                                              are queued for inference servers (see ../inference)
    INDEX_STREAM        scenes:index          Redis stream the scenes are queued on
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import time
from contextlib import chdir
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from aiohttp import web
from livekit import api, rtc

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
PUBLIC_PORT = int(os.environ.get("LIVEKIT_PUBLIC_PORT", "7880"))
ROOM_NAME = os.environ.get("ROOM_NAME", "glasses")
TOKEN_PORT = int(os.environ.get("TOKEN_PORT", "8765"))
ANALYZE_FPS = float(os.environ.get("ANALYZE_FPS", "2"))
YOLOE_MODEL = os.environ.get("YOLOE_MODEL", "yoloe-26s-seg.pt")
YOLOE_CONF = float(os.environ.get("YOLOE_CONF", "0.35"))
FORGET_AFTER_S = float(os.environ.get("FORGET_AFTER_S", "30"))
LEFT_VIEW_S = float(os.environ.get("LEFT_VIEW_S", "3"))
SNAPSHOT_EVERY_S = float(os.environ.get("SNAPSHOT_EVERY_S", "20"))
SNAPSHOT_MIN_CHANGE = float(os.environ.get("SNAPSHOT_MIN_CHANGE", "0.08"))
# Everyday things people misplace. Any text works: YOLOE is open-vocabulary.
DEFAULT_CLASSES = (
    "keys,wallet,phone,glasses,sunglasses,earbuds case,headphones,watch,backpack,"
    "handbag,laptop,tablet,water bottle,umbrella,remote control,credit card,notebook,pen"
)
YOLOE_CLASSES = os.environ.get("YOLOE_CLASSES", DEFAULT_CLASSES)
MODELS_DIR = Path(__file__).parent / "models"
AGENT_IDENTITY = "agent"
REDIS_URL = os.environ.get("REDIS_URL", "")
INDEX_STREAM = os.environ.get("INDEX_STREAM", "scenes:index")
INDEX_STREAM_MAXLEN = 10_000  # cap so an idle queue can't grow without bound

# Initialized in main(); scene processing requires Redis so frames aren't dropped.
redis_client = None

# identity -> most recent location sample from that phone.
latest_location: dict[str, dict] = {}

log = logging.getLogger("live_agent")


# --------------------------------------------------------------------------
# The model: YOLOE open-vocabulary detection. Swap in any other model here.
# --------------------------------------------------------------------------


@dataclass
class Event:
    kind: str  # "arrived" | "left" | "snapshot"
    title: str
    detail: str = ""
    objects: list[dict] = field(default_factory=list)


@dataclass
class Scene:
    """An event plus the frame to save with it (not always the latest frame)."""

    event: Event
    rgb: np.ndarray
    ts_ms: int


class YoloeDetector:
    """Open-vocabulary detection with YOLOE. Shared by all video streams."""

    def __init__(self) -> None:
        import torch
        from ultralytics import YOLOE

        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        classes = [c.strip() for c in YOLOE_CLASSES.split(",") if c.strip()]
        prompt_free = YOLOE_CLASSES.strip() in ("", "*")
        weights = YOLOE_MODEL.replace("-seg.pt", "-seg-pf.pt") if prompt_free and "-pf" not in YOLOE_MODEL else YOLOE_MODEL

        # Weights and the text encoder download into backend/models/ on first run.
        MODELS_DIR.mkdir(exist_ok=True)
        with chdir(MODELS_DIR):
            self.model = YOLOE(weights)
            if not prompt_free:
                self.model.set_classes(classes)
            # Warm up so the first real frame isn't slow.
            self.model.predict(np.zeros((640, 360, 3), np.uint8), device=self.device, verbose=False)
        log.info("YOLOE %s on %s, looking for: %s", weights, self.device,
                 "everything (prompt-free)" if prompt_free else ", ".join(classes))

    def detect(self, rgb: np.ndarray) -> dict[str, dict]:
        """RGB uint8 (height, width, 3) -> best detection per label."""
        result = self.model.predict(rgb[..., ::-1], device=self.device, conf=YOLOE_CONF, verbose=False)[0]
        height, width = rgb.shape[:2]
        best: dict[str, dict] = {}
        for cls, conf, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            label = result.names[int(cls)]
            if label not in best or conf > best[label]["confidence"]:
                x1, y1, x2, y2 = xyxy
                best[label] = {
                    "label": label,
                    "confidence": round(conf, 2),
                    # Normalized [x, y, w, h], origin top-left.
                    "box": [round(x1 / width, 3), round(y1 / height, 3),
                            round((x2 - x1) / width, 3), round((y2 - y1) / height, 3)],
                }
        return best


@dataclass
class _Seen:
    """A frame an object was detected in."""

    rgb: np.ndarray
    ts_ms: int
    detections: dict[str, dict]


def _thumbnail(rgb: np.ndarray, size: int = 16) -> np.ndarray:
    """16x16 block-averaged grayscale, brightness removed, for cheap 'did the view change?'."""
    height, width = rgb.shape[:2]
    bh, bw = max(1, height // size), max(1, width // size)
    gray = rgb[: bh * size, : bw * size].mean(axis=2).reshape(size, bh, size, bw).mean(axis=(1, 3)) / 255
    return gray - gray.mean()


class SightingTracker:
    """Turns per-frame detections from one video stream into scenes worth queuing.

    arrived:  an object detected in two consecutive analyzed frames (filters
              one-frame false positives), not again until it's been out of view
              for FORGET_AFTER_S.
    left:     an arrived object unseen for LEFT_VIEW_S. Queued with the last frame
              it was detected in, which shows where it was put down. Objects that
              left from the same frame share one scene.
    snapshot: at most every SNAPSHOT_EVERY_S, when the view differs enough from
              the last queued scene, regardless of detections.
    """

    def __init__(self) -> None:
        self._pending: set[str] = set()         # seen in the previous frame, not yet announced
        self._last_seen: dict[str, float] = {}  # label -> time last detected
        self._announced: set[str] = set()       # announced, not yet forgotten
        self._prev_visible: set[str] = set()
        self._in_view: dict[str, _Seen] = {}     # announced and in view -> last frame it was in
        self._snapshot_ref: np.ndarray | None = None
        self._snapshot_at = float("-inf")  # snapshot the first frame

    def update(self, rgb: np.ndarray, detections: dict[str, dict], now: float, ts_ms: int) -> list[Scene]:
        """rgb must not be modified afterwards: it may be kept for a later 'left' scene."""
        scenes: list[Scene] = []
        visible = set(detections)
        for label in visible:
            self._last_seen[label] = now
        # Forget things that have been out of view long enough to re-announce.
        for label in list(self._announced):
            if now - self._last_seen.get(label, 0) > FORGET_AFTER_S:
                self._announced.discard(label)

        confirmed = (visible & self._pending) - self._announced
        self._pending = visible - self._announced
        if confirmed:
            self._announced |= confirmed
            objects = sorted((detections[label] for label in confirmed), key=lambda o: -o["confidence"])
            scenes.append(Scene(_event("arrived", "Seen", objects), rgb, ts_ms))

        # Remember the latest frame each announced object is in. Needing two frames
        # in a row (or already being in view) keeps a one-frame blip from moving it.
        seen = _Seen(rgb, ts_ms, detections)
        for label in visible & self._announced:
            if label in self._in_view or label in self._prev_visible:
                self._in_view[label] = seen
        self._prev_visible = visible

        gone = [label for label in self._in_view if now - self._last_seen[label] > LEFT_VIEW_S]
        scenes += self._departures(gone)

        if not scenes and SNAPSHOT_EVERY_S > 0 and now - self._snapshot_at >= SNAPSHOT_EVERY_S:
            thumb = _thumbnail(rgb)
            ref = self._snapshot_ref
            if ref is None or ref.shape != thumb.shape or float(np.abs(thumb - ref).mean()) >= SNAPSHOT_MIN_CHANGE:
                objects = sorted(detections.values(), key=lambda o: -o["confidence"])
                scenes.append(Scene(_event("snapshot", "Snapshot", objects), rgb, ts_ms))
        # Any scene of the current view counts as covering it for snapshots.
        if any(scene.rgb is rgb for scene in scenes):
            self._snapshot_at = now
            self._snapshot_ref = _thumbnail(rgb)
        return scenes

    def flush(self) -> list[Scene]:
        """The stream ended: everything still in view left from its last frame."""
        return self._departures(list(self._in_view))

    def _departures(self, labels: list[str]) -> list[Scene]:
        by_frame: dict[int, tuple[_Seen, list[str]]] = {}
        for label in labels:
            seen = self._in_view.pop(label)
            by_frame.setdefault(id(seen), (seen, []))[1].append(label)
        scenes = []
        for seen, frame_labels in by_frame.values():
            objects = sorted((seen.detections[label] for label in frame_labels), key=lambda o: -o["confidence"])
            scenes.append(Scene(_event("left", "Last seen", objects), seen.rgb, seen.ts_ms))
        return scenes


def _event(kind: str, verb: str, objects: list[dict]) -> Event:
    names = ", ".join(o["label"] for o in objects) or "nothing detected"
    return Event(
        kind=kind,
        title=f"{verb}: {names}",
        detail=", ".join(f"{o['label']} {o['confidence']:.0%}" for o in objects),
        objects=objects,
    )


# --------------------------------------------------------------------------
# Token endpoint
# --------------------------------------------------------------------------


def make_token(identity: str, can_publish: bool) -> str:
    return (
        api.AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_grants(api.VideoGrants(room_join=True, room=ROOM_NAME, can_publish=can_publish))
        .to_jwt()
    )


async def token_handler(request: web.Request) -> web.Response:
    identity = request.query.get("identity") or f"phone-{int(time.time())}"
    # Hand back the LiveKit address on the same host the phone used to reach us,
    # so the phone only needs to know one address.
    host = request.host.rsplit(":", 1)[0]
    body = {
        "url": f"ws://{host}:{PUBLIC_PORT}",
        "token": make_token(identity, can_publish=True),
        "room": ROOM_NAME,
    }
    log.info("issued token for %s", identity)
    return web.json_response(body)


async def location_handler(request: web.Request) -> web.Response:
    try:
        body = await request.json()
        sample = {
            "identity": str(body["identity"]),
            "lat": float(body["lat"]),
            "lon": float(body["lon"]),
            "accuracy_m": float(body.get("accuracy_m", -1)),
            "ts_ms": int(body.get("ts_ms") or time.time() * 1000),
        }
    except (ValueError, KeyError, TypeError):
        return web.json_response({"error": "expected identity, lat, lon"}, status=400)
    if not (-90 <= sample["lat"] <= 90 and -180 <= sample["lon"] <= 180):
        return web.json_response({"error": "lat/lon out of range"}, status=400)

    latest_location[sample["identity"]] = sample
    log.info("location from %s: %.6f, %.6f (±%.0f m)", sample["identity"], sample["lat"], sample["lon"],
             sample["accuracy_m"])
    return web.json_response({"ok": True})


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


async def watch_track(room: rtc.Room, track: rtc.RemoteVideoTrack, participant: rtc.RemoteParticipant,
                      detector: YoloeDetector) -> None:
    stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
    tracker = SightingTracker()
    min_interval = 1.0 / ANALYZE_FPS
    next_at = 0.0
    received = analyzed = 0
    last_log = time.monotonic()
    log.info("watching video from %s", participant.identity)
    try:
        async for frame_event in stream:
            received += 1
            now = time.monotonic()
            if now - last_log >= 5:
                log.info("%s: %.1f fps in, %.1f fps analyzed", participant.identity,
                         received / (now - last_log), analyzed / (now - last_log))
                received = analyzed = 0
                last_log = now
            if now < next_at:
                continue
            next_at = now + min_interval
            analyzed += 1

            frame = frame_event.frame
            # Copy: the tracker may hold on to this frame for a later "left" scene.
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()
            ts_ms = int(time.time() * 1000)
            # Inference runs off the event loop so LiveKit keeps flowing.
            detections = await asyncio.to_thread(detector.detect, rgb)
            for scene in tracker.update(rgb, detections, time.monotonic(), ts_ms):
                asyncio.create_task(save_and_enqueue(scene, participant.identity))
                if scene.event.kind == "arrived":
                    asyncio.create_task(send_alert(room, participant.identity, scene.event))
    finally:
        for scene in tracker.flush():
            asyncio.create_task(save_and_enqueue(scene, participant.identity))
        await stream.aclose()
        log.info("stopped watching %s", participant.identity)


def encode_scene(scene: Scene, identity: str) -> tuple[dict, bytes]:
    """Encode a selected frame in memory and return its Redis job and JPEG bytes."""
    from PIL import Image

    ts_ms = scene.ts_ms  # when the frame was captured, not now ("left" frames are older)
    # kind in the name: an "arrived" and a "left" scene can share one frame.
    scene_id = f"{ts_ms}_{identity}_{scene.event.kind}"
    buf = io.BytesIO()
    Image.fromarray(scene.rgb).save(buf, format="JPEG", quality=90)
    jpeg = buf.getvalue()
    record = {"id": scene_id, **asdict(scene.event), "ts_ms": ts_ms, "identity": identity,
              "image": f"{scene_id}.jpg"}
    # The phone only reports when it moves, so the last sample still applies while
    # standing still. age_s says how old it is.
    if (loc := latest_location.get(identity)) is not None:
        record["location"] = {**loc, "age_s": round((ts_ms - loc["ts_ms"]) / 1000, 1)}
    log.info("encoded scene %s (%s)", scene_id, scene.event.title)
    return record, jpeg


async def save_and_enqueue(scene: Scene, identity: str) -> None:
    record, jpeg = await asyncio.to_thread(encode_scene, scene, identity)
    # The image rides along in the job, so inference servers need no shared disk.
    job = {**record, "image_b64": base64.b64encode(jpeg).decode("ascii")}
    try:
        if redis_client is None:
            raise RuntimeError("Redis is not initialized")
        await redis_client.xadd(INDEX_STREAM, {"job": json.dumps(job)},
                                maxlen=INDEX_STREAM_MAXLEN, approximate=True)
        log.info("queued scene %s on %s", record["id"], INDEX_STREAM)
    except Exception as exc:  # Keep detection alive; this frame is intentionally not persisted locally.
        log.error("could not queue scene %s; frame was not saved locally: %s", record["id"], exc)


async def send_alert(room: rtc.Room, identity: str, event: Event) -> None:
    payload = json.dumps({**asdict(event), "ts_ms": int(time.time() * 1000)})
    try:
        reply = await room.local_participant.perform_rpc(
            destination_identity=identity, method="alert", payload=payload, response_timeout=5.0
        )
        log.info("alert -> %s: %s (reply %s)", identity, event.title, reply)
    except Exception as exc:  # phone gone or not listening
        log.warning("alert to %s failed: %s", identity, exc)


async def run_agent() -> None:
    detector = YoloeDetector()
    room = rtc.Room()
    tasks: dict[str, asyncio.Task] = {}

    @room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication,
                            participant: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            tasks[publication.sid] = asyncio.create_task(watch_track(room, track, participant, detector))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication,
                              participant: rtc.RemoteParticipant) -> None:
        task = tasks.pop(publication.sid, None)
        if task:
            task.cancel()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        log.info("joined: %s", participant.identity)

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant) -> None:
        log.info("left: %s", participant.identity)

    await room.connect(LIVEKIT_URL, make_token(AGENT_IDENTITY, can_publish=False))
    log.info("agent joined room %r at %s", ROOM_NAME, LIVEKIT_URL)
    await asyncio.Future()


async def main() -> None:
    global redis_client
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required; selected frames are queued directly and are not saved locally")
    import redis.asyncio as redis

    redis_client = redis.from_url(REDIS_URL)
    await redis_client.ping()
    from urllib.parse import urlsplit

    # Don't log the URL itself: it may carry the Redis password.
    log.info("queueing scenes on redis %s stream %r", urlsplit(REDIS_URL).hostname, INDEX_STREAM)
    app = web.Application()
    app.router.add_get("/token", token_handler)
    app.router.add_post("/location", location_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", TOKEN_PORT).start()
    log.info("token server on http://0.0.0.0:%d/token", TOKEN_PORT)
    await run_agent()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    asyncio.run(main())
