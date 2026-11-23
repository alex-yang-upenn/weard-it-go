import asyncio
import json
import time

from livekit import rtc
from common.raw_frames.raw_frame_archive import RawFrameArchive
from common.raw_frames.raw_frame_codec import compress_raw_frame
from common.redis_streams.sampled_frame_message import SampledFrameMessage

from outputs.frame_metadata_repository import FrameMetadataRepository
from recording.latest_telemetry_tracker import LatestTelemetryTracker
from outputs.sampled_frame_publisher import SampledFramePublisher
from recording.uniform_frame_sampler import UniformFrameSampler


def current_time_ms() -> int:
    return int(time.time() * 1000)


class RoomFrameRecorder:
    def __init__(
        self,
        session_id: str,
        user_id: str,
        frame_archive: RawFrameArchive,
        frame_metadata_repository: FrameMetadataRepository,
        sampled_frame_publisher: SampledFramePublisher,
        uniform_frame_sampler: UniformFrameSampler,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.frame_archive = frame_archive
        self.frame_metadata_repository = frame_metadata_repository
        self.sampled_frame_publisher = sampled_frame_publisher
        self.uniform_frame_sampler = uniform_frame_sampler
        self.latest_telemetry_tracker = LatestTelemetryTracker()
        self.received_frame_count = 0
        self.open_video_streams: list[rtc.VideoStream] = []

    async def record_video_track(self, video_track: rtc.RemoteVideoTrack) -> None:
        video_stream = rtc.VideoStream(video_track)
        self.open_video_streams.append(video_stream)
        async for frame_event in video_stream:
            await self.record_frame(frame_event.frame)

    async def stop_recording_video(self) -> None:
        for video_stream in self.open_video_streams:
            await video_stream.aclose()

    async def record_frame(self, frame: rtc.VideoFrame) -> None:
        received_at_ms = current_time_ms()
        frame_index = self.received_frame_count
        self.received_frame_count += 1
        frame_id = f"{self.session_id}-{frame_index}"
        compressed_i420_bytes = await asyncio.to_thread(compress_raw_frame, bytes(frame.data))
        file_path = await asyncio.to_thread(
            self.frame_archive.save_frame, self.session_id, frame_index, compressed_i420_bytes
        )
        frame_metadata = self.latest_telemetry_tracker.build_frame_metadata()
        await self.frame_metadata_repository.insert_frame(
            frame_id, self.session_id, self.user_id, frame_index, received_at_ms,
            frame.width, frame.height, file_path, frame_metadata,
        )
        if self.uniform_frame_sampler.should_sample(received_at_ms):
            await self.sampled_frame_publisher.publish_sampled_frame(
                SampledFrameMessage(
                    frame_id=frame_id,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    received_at_ms=received_at_ms,
                    width=frame.width,
                    height=frame.height,
                    compressed_i420_bytes=compressed_i420_bytes,
                    metadata=frame_metadata,
                )
            )

    async def record_telemetry(self, telemetry_json_bytes: bytes) -> None:
        received_at_ms = current_time_ms()
        telemetry = json.loads(telemetry_json_bytes)
        self.latest_telemetry_tracker.record(telemetry, received_at_ms)
        await self.frame_metadata_repository.insert_telemetry(self.session_id, received_at_ms, telemetry)

    async def finish_session(self) -> None:
        await self.frame_metadata_repository.mark_session_ended(self.session_id)
        await self.sampled_frame_publisher.publish_session_ended(self.session_id)
