"""Queue scenes already on disk (backend/scenes) for indexing.

Useful to backfill, or to test the inference server without the glasses:
    pip install redis
    REDIS_URL=redis://:<password>@<host>:6379 python inference/enqueue_scenes.py
    python inference/enqueue_scenes.py --limit 5 backend/scenes

Jobs have the same shape live_agent.py sends: the scene JSON plus image_b64.
Re-queuing a scene is harmless; the turbopuffer row is keyed by the scene id.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import redis

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "backend" / "scenes"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes_dir", nargs="?", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379"))
    parser.add_argument("--stream", default=os.environ.get("STREAM", "scenes:index"))
    parser.add_argument("--limit", type=int, default=0, help="queue at most this many (newest first)")
    args = parser.parse_args()

    files = sorted(args.scenes_dir.glob("*.json"), reverse=True)
    if args.limit:
        files = files[: args.limit]
    r = redis.from_url(args.redis_url)
    queued = skipped = 0
    for path in files:
        scene = json.loads(path.read_text())
        image = args.scenes_dir / scene.get("image", "")
        if not image.is_file():
            print(f"skip {path.name}: no image")
            skipped += 1
            continue
        job = {**scene, "image_b64": base64.b64encode(image.read_bytes()).decode("ascii")}
        r.xadd(args.stream, {"job": json.dumps(job)}, maxlen=10_000, approximate=True)
        queued += 1
    print(f"queued {queued} scenes on {args.stream!r} ({skipped} skipped); stream length {r.xlen(args.stream)}")


if __name__ == "__main__":
    main()
