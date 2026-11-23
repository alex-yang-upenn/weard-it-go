import json

from livekit import api


class LivekitRoomManager:
    def __init__(self, livekit_api_url: str, livekit_public_url: str, livekit_api_key: str, livekit_api_secret: str):
        self.livekit_public_url = livekit_public_url
        self.livekit_api_key = livekit_api_key
        self.livekit_api_secret = livekit_api_secret
        self.livekit_api = api.LiveKitAPI(livekit_api_url, livekit_api_key, livekit_api_secret)

    async def create_room_for_user(self, room_name: str, user_id: str) -> None:
        await self.livekit_api.room.create_room(
            api.CreateRoomRequest(name=room_name, metadata=json.dumps({"user_id": user_id}))
        )

    def create_participant_token(self, room_name: str, user_id: str) -> str:
        return (
            api.AccessToken(self.livekit_api_key, self.livekit_api_secret)
            .with_identity(user_id)
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )

    async def close(self) -> None:
        await self.livekit_api.aclose()
