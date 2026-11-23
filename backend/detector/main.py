import asyncio
import logging
import os

import redis.asyncio as redis
import torch

from common.redis_streams.redis_stream_consumer import RedisStreamConsumer
from common.redis_streams.sampled_frame_message import SAMPLED_FRAMES_STREAM
from detection.sampled_frame_batch_handler import SampledFrameBatchHandler
from detection.yoloe_object_detector import YoloeObjectDetector
from scene_selection.scene_publisher import ScenePublisher
from scene_selection.scene_selector import SceneSelector

MAX_FRAMES_PER_BATCH = 16

logger = logging.getLogger(__name__)


def read_snapshot_interval_seconds() -> float | None:
    snapshot_interval = os.environ.get("SNAPSHOT_INTERVAL_S")
    return float(snapshot_interval) if snapshot_interval else None


async def run_detector() -> None:
    redis_client = redis.from_url(os.environ["REDIS_URL"])
    sampled_frame_consumer = RedisStreamConsumer(redis_client, SAMPLED_FRAMES_STREAM, "detector", "detector")
    await sampled_frame_consumer.create_group_if_missing()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_detector = YoloeObjectDetector(os.environ["MODEL_DIRECTORY"], device)
    logger.info(f"YOLOE loaded on {device}, waiting for sampled frames")
    batch_handler = SampledFrameBatchHandler(
        object_detector=object_detector,
        scene_selector=SceneSelector(read_snapshot_interval_seconds()),
        scene_publisher=ScenePublisher(redis_client),
    )
    while True:
        stream_messages = await sampled_frame_consumer.read_batch(MAX_FRAMES_PER_BATCH)
        if not stream_messages:
            continue
        await batch_handler.handle_batch(stream_messages)
        await sampled_frame_consumer.acknowledge([message_id for message_id, _ in stream_messages])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    asyncio.run(run_detector())
