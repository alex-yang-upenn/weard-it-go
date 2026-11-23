import asyncpg
from common.postgres_connection_pool import format_vector_for_pgvector


class FrameSearchRepository:
    def __init__(self, connection_pool: asyncpg.Pool):
        self.connection_pool = connection_pool

    async def find_frames_closest_to_embedding(
        self, user_id: str, query_embedding: list[float], max_cosine_distance: float, result_limit: int
    ) -> list[dict]:
        rows = await self.connection_pool.fetch(
            "SELECT embeddings.frame_id, embeddings.embedding <=> $1::vector AS cosine_distance, "
            "frames.session_id, frames.received_at_ms, frames.metadata "
            "FROM embeddings JOIN frames USING (frame_id) "
            "WHERE embeddings.user_id = $2 AND embeddings.embedding <=> $1::vector < $3 "
            "ORDER BY cosine_distance LIMIT $4",
            format_vector_for_pgvector(query_embedding), user_id, max_cosine_distance, result_limit,
        )
        return [dict(row) for row in rows]

    async def find_frame_file(self, frame_id: str) -> asyncpg.Record | None:
        return await self.connection_pool.fetchrow(
            "SELECT file_path, width, height FROM frames WHERE frame_id = $1", frame_id
        )
