import redis.asyncio as redis
from common.redis_streams.sampled_frame_message import (
    SAMPLED_FRAMES_STREAM,
    SampledFrameMessage,
    build_session_ended_fields,
)


class SampledFramePublisher:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def publish_sampled_frame(self, sampled_frame: SampledFrameMessage) -> None:
        await self.redis_client.xadd(
            SAMPLED_FRAMES_STREAM, sampled_frame.to_stream_fields(), maxlen=1000, approximate=True
        )

    async def publish_session_ended(self, session_id: str) -> None:
        await self.redis_client.xadd(SAMPLED_FRAMES_STREAM, build_session_ended_fields(session_id))
