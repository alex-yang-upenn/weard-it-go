import asyncpg


class SessionRepository:
    def __init__(self, connection_pool: asyncpg.Pool):
        self.connection_pool = connection_pool

    async def insert_session(self, session_id: str, user_id: str) -> None:
        await self.connection_pool.execute(
            "INSERT INTO sessions (session_id, user_id) VALUES ($1, $2)", session_id, user_id
        )
