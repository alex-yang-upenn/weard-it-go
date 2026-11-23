import json

import asyncpg


async def use_json_for_jsonb_columns(connection: asyncpg.Connection) -> None:
    await connection.set_type_codec("jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog")


async def create_postgres_connection_pool(database_url: str) -> asyncpg.Pool:
    return await asyncpg.create_pool(database_url, min_size=1, max_size=4, init=use_json_for_jsonb_columns)


def format_vector_for_pgvector(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"
