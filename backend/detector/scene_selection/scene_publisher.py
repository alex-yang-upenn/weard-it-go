import time
import uuid

import redis.asyncio as redis
from common.raw_frames.raw_frame_codec import encode_rgb_as_jpeg
from common.redis_streams.scene_message import SCENES_STREAM, SceneMessage

from scene_selection.object_sighting_tracker import SelectedScene
from scene_selection.queued_scene_logger import log_queued_scene


class ScenePublisher:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    async def publish_scene(self, selected_scene: SelectedScene) -> None:
        sampled_frame = selected_scene.detected_frame.sampled_frame
        scene_message = SceneMessage(
            scene_id=uuid.uuid4().hex,
            frame_id=sampled_frame.frame_id,
            session_id=sampled_frame.session_id,
            user_id=sampled_frame.user_id,
            kind=selected_scene.kind,
            objects=selected_scene.objects,
            created_at_ms=int(time.time() * 1000),
            jpeg_bytes=encode_rgb_as_jpeg(selected_scene.detected_frame.rgb),
        )
        await self.redis_client.xadd(SCENES_STREAM, scene_message.to_stream_fields(), maxlen=1000, approximate=True)
        log_queued_scene(selected_scene)
