import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.frame_routes import frame_router
from api.search_routes import search_router
from api.session_routes import session_router
from common.postgres_connection_pool import create_postgres_connection_pool
from common.raw_frames.raw_frame_archive import RawFrameArchive
from database.frame_search_repository import FrameSearchRepository
from database.session_repository import SessionRepository
from integrations.embedder_client import EmbedderClient
from integrations.livekit_room_manager import LivekitRoomManager


@asynccontextmanager
async def open_service_connections(app: FastAPI):
    connection_pool = await create_postgres_connection_pool(os.environ["DATABASE_URL"])
    app.state.session_repository = SessionRepository(connection_pool)
    app.state.frame_search_repository = FrameSearchRepository(connection_pool)
    app.state.livekit_room_manager = LivekitRoomManager(
        livekit_api_url=os.environ["LIVEKIT_API_URL"],
        livekit_public_url=os.environ["LIVEKIT_PUBLIC_URL"],
        livekit_api_key=os.environ["LIVEKIT_API_KEY"],
        livekit_api_secret=os.environ["LIVEKIT_API_SECRET"],
    )
    app.state.embedder_client = EmbedderClient(os.environ["EMBEDDER_URL"])
    app.state.frame_archive = RawFrameArchive(os.environ["FRAME_ARCHIVE_DIRECTORY"])
    yield
    await app.state.livekit_room_manager.close()
    await connection_pool.close()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=open_service_connections)
    app.include_router(session_router)
    app.include_router(search_router)
    app.include_router(frame_router)
    return app
