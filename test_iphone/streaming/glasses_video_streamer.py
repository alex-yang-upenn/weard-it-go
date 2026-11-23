import asyncio
import time

import av
from livekit import rtc

from streaming.fake_gps_publisher import FakeGpsPublisher
from streaming.video_segment import VideoSegment

GLASSES_CAMERA_WIDTH = 1280
GLASSES_CAMERA_HEIGHT = 720


async def publish_segment_in_real_time(video_source: rtc.VideoSource, segment: VideoSegment) -> None:
    with av.open(str(segment.video_path)) as container:
        video_stream = container.streams.video[0]
        container.seek(int(segment.start_seconds / video_stream.time_base), stream=video_stream)
        segment_started_at = time.monotonic()
        for decoded_frame in container.decode(video_stream):
            seconds_into_segment = decoded_frame.time - segment.start_seconds
            if seconds_into_segment < 0:
                continue
            if seconds_into_segment >= segment.duration_seconds:
                break
            await asyncio.sleep(max(0.0, segment_started_at + seconds_into_segment - time.monotonic()))
            i420_bytes = decoded_frame.to_ndarray(format="yuv420p").tobytes()
            video_source.capture_frame(
                rtc.VideoFrame(decoded_frame.width, decoded_frame.height, rtc.VideoBufferType.I420, i420_bytes)
            )


class GlassesVideoStreamer:
    async def stream(self, livekit_url: str, token: str, segments: list[VideoSegment]) -> None:
        room = rtc.Room()
        await room.connect(livekit_url, token)
        video_source = rtc.VideoSource(GLASSES_CAMERA_WIDTH, GLASSES_CAMERA_HEIGHT)
        await room.local_participant.publish_track(
            rtc.LocalVideoTrack.create_video_track("glasses-camera", video_source),
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
        )
        gps_task = asyncio.create_task(FakeGpsPublisher(room).publish_every_second())
        for segment in segments:
            await publish_segment_in_real_time(video_source, segment)
        gps_task.cancel()
        await room.disconnect()
