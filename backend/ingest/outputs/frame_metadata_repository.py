import asyncpg


class FrameMetadataRepository:
    def __init__(self, connection_pool: asyncpg.Pool):
        self.connection_pool = connection_pool

    async def insert_frame(
        self,
        frame_id: str,
        session_id: str,
        user_id: str,
        frame_index: int,
        received_at_ms: int,
        width: int,
        height: int,
        file_path: str,
        metadata: dict | None,
    ) -> None:
        await self.connection_pool.execute(
            "INSERT INTO frames (frame_id, session_id, user_id, frame_index, received_at_ms, width, height, file_path, metadata) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
            frame_id, session_id, user_id, frame_index, received_at_ms, width, height, file_path, metadata,
        )

    async def insert_telemetry(self, session_id: str, received_at_ms: int, payload: dict) -> None:
        await self.connection_pool.execute(
            "INSERT INTO telemetry (session_id, received_at_ms, payload) VALUES ($1, $2, $3)",
            session_id, received_at_ms, payload,
        )

    async def mark_session_ended(self, session_id: str) -> None:
        await self.connection_pool.execute("UPDATE sessions SET ended_at = now() WHERE session_id = $1", session_id)
