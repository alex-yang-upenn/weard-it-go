import asyncio
import json
import logging
import os

import redis.asyncio as redis
from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, JobProcess, WorkerOptions, cli

from common.postgres_connection_pool import create_postgres_connection_pool
from common.raw_frames.raw_frame_archive import RawFrameArchive
from outputs.frame_metadata_repository import FrameMetadataRepository
from outputs.sampled_frame_publisher import SampledFramePublisher
from recording.room_frame_recorder import RoomFrameRecorder
from recording.uniform_frame_sampler import UniformFrameSampler

SAMPLED_FRAMES_PER_SECOND = 8

logger = logging.getLogger("ingest")


def configure_job_process_logging(job_process: JobProcess) -> None:
    logging.getLogger("livekit").setLevel(logging.ERROR)
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(log_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


async def record_room_until_phone_leaves(job_context: JobContext) -> None:
    connection_pool = await create_postgres_connection_pool(os.environ["DATABASE_URL"])
    redis_client = redis.from_url(os.environ["REDIS_URL"])
    room = job_context.room
    recorder = RoomFrameRecorder(
        session_id=job_context.job.room.name,
        user_id=json.loads(job_context.job.room.metadata)["user_id"],
        frame_archive=RawFrameArchive(os.environ["FRAME_ARCHIVE_DIRECTORY"]),
        frame_metadata_repository=FrameMetadataRepository(connection_pool),
        sampled_frame_publisher=SampledFramePublisher(redis_client),
        uniform_frame_sampler=UniformFrameSampler(SAMPLED_FRAMES_PER_SECOND),
    )
    running_tasks: set[asyncio.Task] = set()
    video_recording_tasks: list[asyncio.Task] = []
    phone_left_room = asyncio.Event()

    def keep_task_alive(task: asyncio.Task) -> None:
        running_tasks.add(task)
        task.add_done_callback(running_tasks.discard)

    @room.on("track_subscribed")
    def start_recording_video(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            video_recording_task = asyncio.create_task(recorder.record_video_track(track))
            video_recording_tasks.append(video_recording_task)
            keep_task_alive(video_recording_task)

    @room.on("data_received")
    def store_telemetry(data_packet: rtc.DataPacket):
        if data_packet.topic == "telemetry":
            keep_task_alive(asyncio.create_task(recorder.record_telemetry(data_packet.data)))

    @room.on("participant_disconnected")
    def notice_phone_left(participant: rtc.RemoteParticipant):
        phone_left_room.set()

    await job_context.connect(auto_subscribe=AutoSubscribe.VIDEO_ONLY)
    logger.info(f"recording session {recorder.session_id} for user {recorder.user_id}")
    await phone_left_room.wait()
    for video_recording_task in video_recording_tasks:
        video_recording_task.cancel()
    await asyncio.gather(*video_recording_tasks, return_exceptions=True)
    await recorder.stop_recording_video()
    await recorder.finish_session()
    logger.info(f"session {recorder.session_id} ended after {recorder.received_frame_count} frames")
    await connection_pool.close()
    await redis_client.aclose()
    job_context.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=record_room_until_phone_leaves,
            prewarm_fnc=configure_job_process_logging,
            load_threshold=float("inf"),
        )
    )
