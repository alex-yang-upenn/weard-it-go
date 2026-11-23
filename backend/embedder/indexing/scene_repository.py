import asyncpg
from common.postgres_connection_pool import format_vector_for_pgvector
from common.redis_streams.scene_message import SceneMessage


class SceneRepository:
    def __init__(self, connection_pool: asyncpg.Pool):
        self.connection_pool = connection_pool

    async def insert_scene(self, scene: SceneMessage) -> None:
        await self.connection_pool.execute(
            "INSERT INTO scenes (scene_id, frame_id, session_id, user_id, kind, objects, created_at_ms) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) ON CONFLICT DO NOTHING",
            scene.scene_id, scene.frame_id, scene.session_id, scene.user_id, scene.kind, scene.objects,
            scene.created_at_ms,
        )

    async def insert_frame_embedding(self, frame_id: str, user_id: str, embedding: list[float]) -> None:
        await self.connection_pool.execute(
            "INSERT INTO embeddings (frame_id, user_id, embedding) VALUES ($1, $2, $3::vector) ON CONFLICT DO NOTHING",
            frame_id, user_id, format_vector_for_pgvector(embedding),
        )
