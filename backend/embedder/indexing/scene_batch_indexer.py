import asyncio
import logging
import time

from common.redis_streams.redis_stream_consumer import RedisStreamConsumer
from common.redis_streams.scene_message import SceneMessage
from model.qwen_multimodal_embedder import QwenMultimodalEmbedder
from indexing.scene_repository import SceneRepository

MAX_SCENES_PER_BATCH = 16

logger = logging.getLogger(__name__)


class SceneBatchIndexer:
    def __init__(
        self,
        scene_consumer: RedisStreamConsumer,
        multimodal_embedder: QwenMultimodalEmbedder,
        scene_repository: SceneRepository,
    ):
        self.scene_consumer = scene_consumer
        self.multimodal_embedder = multimodal_embedder
        self.scene_repository = scene_repository

    async def index_scenes_forever(self) -> None:
        await self.scene_consumer.create_group_if_missing()
        while True:
            stream_messages = await self.scene_consumer.read_batch(MAX_SCENES_PER_BATCH)
            if not stream_messages:
                continue
            await self.index_scenes([SceneMessage.from_stream_fields(fields) for _, fields in stream_messages])
            await self.scene_consumer.acknowledge([message_id for message_id, _ in stream_messages])

    async def index_scenes(self, scenes: list[SceneMessage]) -> None:
        started_at = time.monotonic()
        for scene in scenes:
            await self.scene_repository.insert_scene(scene)
        unique_frame_scenes = list({scene.frame_id: scene for scene in scenes}.values())
        embeddings = await asyncio.to_thread(
            self.multimodal_embedder.embed_jpeg_images, [scene.jpeg_bytes for scene in unique_frame_scenes]
        )
        for scene, embedding in zip(unique_frame_scenes, embeddings):
            await self.scene_repository.insert_frame_embedding(scene.frame_id, scene.user_id, embedding)
        logger.info(f"embedded {len(unique_frame_scenes)} frames for {len(scenes)} scenes in {time.monotonic() - started_at:.1f}s")
