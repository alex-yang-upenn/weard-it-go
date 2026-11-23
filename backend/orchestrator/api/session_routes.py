import uuid

from fastapi import APIRouter, Request
from pydantic import BaseModel


class CreateSessionRequest(BaseModel):
    user_id: str


class CreateSessionResponse(BaseModel):
    session_id: str
    livekit_url: str
    token: str


session_router = APIRouter()


@session_router.post("/sessions")
async def create_session(body: CreateSessionRequest, request: Request) -> CreateSessionResponse:
    services = request.app.state
    session_id = uuid.uuid4().hex
    await services.session_repository.insert_session(session_id, body.user_id)
    await services.livekit_room_manager.create_room_for_user(session_id, body.user_id)
    return CreateSessionResponse(
        session_id=session_id,
        livekit_url=services.livekit_room_manager.livekit_public_url,
        token=services.livekit_room_manager.create_participant_token(session_id, body.user_id),
    )
