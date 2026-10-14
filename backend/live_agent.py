"""Live video agent for Wear'd It Go.

One process that:
  1. Serves access tokens to the iOS app:  GET /token?identity=<id>
     -> {"url": "ws://<this-host>:7880", "token": "...", "room": "glasses"}
  2. Joins the same LiveKit room as "agent", subscribes to the glasses video,
     samples frames, runs YOLOE open-vocabulary detection on them, and pings
     the phone back over LiveKit RPC (method "alert") when objects come into view.

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
"""

from __future__ import annotations

import asyncio
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
# Everyday things people misplace. Any text works: YOLOE is open-vocabulary.
DEFAULT_CLASSES = (
    "keys,wallet,phone,glasses,sunglasses,earbuds case,headphones,watch,backpack,"
    "handbag,laptop,tablet,water bottle,umbrella,remote control,credit card,notebook,pen"
)
YOLOE_CLASSES = os.environ.get("YOLOE_CLASSES", DEFAULT_CLASSES)
MODELS_DIR = Path(__file__).parent / "models"
SCENES_DIR = Path(os.environ.get("SCENES_DIR", Path(__file__).parent / "scenes"))
AGENT_IDENTITY = "agent"

log = logging.getLogger("live_agent")


# --------------------------------------------------------------------------
# The model: YOLOE open-vocabulary detection. Swap in any other model here.
# --------------------------------------------------------------------------


@dataclass
class Event:
    title: str
    detail: str = ""
    objects: list[dict] = field(default_factory=list)


class YoloeDetector:
    """Open-vocabulary detection with YOLOE; reports objects as they come into view.

    Gets RGB uint8 frames of shape (height, width, 3) at ANALYZE_FPS. An object
    is announced once it's been detected in two consecutive analyzed frames
    (filters one-frame false positives), then not again until it's been out of
    view for FORGET_AFTER_S.
    """

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

        self._pending: set[str] = set()       # seen in the previous frame, not yet announced
        self._last_seen: dict[str, float] = {}  # label -> time last detected
        self._announced: set[str] = set()     # currently "in view" and already announced

    def analyze(self, rgb: np.ndarray) -> Event | None:
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

        now = time.monotonic()
        for label in best:
            self._last_seen[label] = now
        # Forget things that have been out of view long enough to re-announce.
        for label in list(self._announced):
            if now - self._last_seen.get(label, 0) > FORGET_AFTER_S:
                self._announced.discard(label)

        visible = set(best)
        confirmed = (visible & self._pending) - self._announced
        self._pending = visible - self._announced
        if not confirmed:
            return None
        self._announced |= confirmed
        objects = sorted((best[label] for label in confirmed), key=lambda o: -o["confidence"])
        names = [o["label"] for o in objects]
        return Event(
            title=f"Seen: {', '.join(names)}",
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


# --------------------------------------------------------------------------
# Agent
# --------------------------------------------------------------------------


async def watch_track(room: rtc.Room, track: rtc.RemoteVideoTrack, participant: rtc.RemoteParticipant,
                      detector: YoloeDetector) -> None:
    stream = rtc.VideoStream(track, format=rtc.VideoBufferType.RGB24)
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
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            # Inference runs off the event loop so LiveKit keeps flowing.
            event = await asyncio.to_thread(detector.analyze, rgb)
            if event is not None:
                asyncio.create_task(asyncio.to_thread(save_scene, rgb.copy(), participant.identity, event))
                asyncio.create_task(send_alert(room, participant.identity, event))
    finally:
        await stream.aclose()
        log.info("stopped watching %s", participant.identity)


def save_scene(rgb: np.ndarray, identity: str, event: Event) -> Path:
    """Write the frame (JPEG) plus detections (JSON) so a VLM can be run on it."""
    from PIL import Image

    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    ts_ms = int(time.time() * 1000)
    stem = SCENES_DIR / f"{ts_ms}_{identity}"
    image_path = stem.with_suffix(".jpg")
    Image.fromarray(rgb).save(image_path, quality=90)
    stem.with_suffix(".json").write_text(
        json.dumps({**asdict(event), "ts_ms": ts_ms, "identity": identity, "image": image_path.name}, indent=2)
    )
    log.info("saved scene %s", image_path)
    return image_path


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
    app = web.Application()
    app.router.add_get("/token", token_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "0.0.0.0", TOKEN_PORT).start()
    log.info("token server on http://0.0.0.0:%d/token", TOKEN_PORT)
    await run_agent()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    asyncio.run(main())
