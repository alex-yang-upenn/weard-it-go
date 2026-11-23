import asyncio

from common.raw_frames.raw_frame_codec import convert_i420_to_rgb, decompress_raw_frame
from common.redis_streams.sampled_frame_message import SampledFrameMessage, is_session_ended_message
from scene_selection.object_sighting_tracker import DetectedFrame
from scene_selection.scene_publisher import ScenePublisher
from scene_selection.scene_selector import SceneSelector
from detection.yoloe_object_detector import YoloeObjectDetector


def decode_sampled_frame_to_rgb(sampled_frame: SampledFrameMessage):
    i420_bytes = decompress_raw_frame(sampled_frame.compressed_i420_bytes)
    return convert_i420_to_rgb(i420_bytes, sampled_frame.width, sampled_frame.height)


class SampledFrameBatchHandler:
    def __init__(self, object_detector: YoloeObjectDetector, scene_selector: SceneSelector, scene_publisher: ScenePublisher):
        self.object_detector = object_detector
        self.scene_selector = scene_selector
        self.scene_publisher = scene_publisher

    async def handle_batch(self, stream_messages: list[tuple[bytes, dict]]) -> None:
        sampled_frames = [
            SampledFrameMessage.from_stream_fields(fields)
            for _, fields in stream_messages
            if not is_session_ended_message(fields)
        ]
        rgb_frames = [decode_sampled_frame_to_rgb(sampled_frame) for sampled_frame in sampled_frames]
        detections_per_frame = await asyncio.to_thread(self.object_detector.detect_objects_in_batch, rgb_frames)
        detected_frames = iter(
            DetectedFrame(sampled_frame, rgb, detections)
            for sampled_frame, rgb, detections in zip(sampled_frames, rgb_frames, detections_per_frame)
        )
        for _, fields in stream_messages:
            if is_session_ended_message(fields):
                selected_scenes = self.scene_selector.end_session(fields[b"session_id"].decode())
            else:
                selected_scenes = self.scene_selector.select_scenes(next(detected_frames))
            for selected_scene in selected_scenes:
                await self.scene_publisher.publish_scene(selected_scene)
