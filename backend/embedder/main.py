import asyncio
import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as redis
from fastapi import FastAPI, Request
from pydantic import BaseModel

from common.postgres_connection_pool import create_postgres_connection_pool
from common.redis_streams.redis_stream_consumer import RedisStreamConsumer
from common.redis_streams.scene_message import SCENES_STREAM
from model.qwen_multimodal_embedder import QwenMultimodalEmbedder
from indexing.scene_batch_indexer import SceneBatchIndexer
from indexing.scene_repository import SceneRepository


logger = logging.getLogger(__name__)


class EmbedTextRequest(BaseModel):
    text: str


@asynccontextmanager
async def load_model_and_start_indexing(app: FastAPI):
    multimodal_embedder = QwenMultimodalEmbedder()
    logger.info("Qwen3-VL embedder loaded, indexing scenes")
    connection_pool = await create_postgres_connection_pool(os.environ["DATABASE_URL"])
    scene_batch_indexer = SceneBatchIndexer(
        scene_consumer=RedisStreamConsumer(redis.from_url(os.environ["REDIS_URL"]), SCENES_STREAM, "embedder", "embedder"),
        multimodal_embedder=multimodal_embedder,
        scene_repository=SceneRepository(connection_pool),
    )
    app.state.multimodal_embedder = multimodal_embedder
    indexing_task = asyncio.create_task(scene_batch_indexer.index_scenes_forever())
    yield
    indexing_task.cancel()


def create_app() -> FastAPI:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = FastAPI(lifespan=load_model_and_start_indexing)

    @app.post("/embed/text")
    def embed_search_text(body: EmbedTextRequest, request: Request) -> dict:
        return {"embedding": request.app.state.multimodal_embedder.embed_search_text(body.text)}

    return app
