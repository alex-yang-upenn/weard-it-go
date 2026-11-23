import argparse
import asyncio
import json
import logging
import math
from pathlib import Path

from backend_api.orchestrator_client import OrchestratorClient
from samples.mixkit_clip_downloader import MixkitClipDownloader
from samples.sample_video_catalog import SAMPLE_VIDEOS, find_sample_video
from streaming.glasses_video_streamer import GlassesVideoStreamer
from streaming.video_segment import VideoSegment

SAMPLE_VIDEO_CACHE_DIRECTORY = Path(__file__).parent / "sample_videos"

logger = logging.getLogger("test_iphone")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Does what the iPhone app does, against the Wear'd It Go backend.")
    parser.add_argument("--orchestrator-url", default="http://localhost:8000")
    parser.add_argument("--user-id", default="test-iphone")
    commands = parser.add_subparsers(dest="command", required=True)

    stream_command = commands.add_parser("stream", help="start a session and stream a video with fake GPS")
    stream_command.add_argument(
        "sample_name_or_video_path", help="an .mp4 path or a sample: " + ", ".join(s.name for s in SAMPLE_VIDEOS)
    )

    search_command = commands.add_parser("search", help="print search results as JSON")
    search_command.add_argument("query")
    search_command.add_argument("--limit", type=int, default=5)

    frame_command = commands.add_parser("frame", help="save a stored frame as a JPEG")
    frame_command.add_argument("frame_id")
    frame_command.add_argument("output_path", nargs="?")
    return parser


def resolve_video_segments(sample_name_or_video_path: str) -> list[VideoSegment]:
    sample = find_sample_video(sample_name_or_video_path)
    if sample is None:
        return [VideoSegment(Path(sample_name_or_video_path), 0, math.inf)]
    return MixkitClipDownloader(SAMPLE_VIDEO_CACHE_DIRECTORY).find_or_download_sample(sample)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("livekit").setLevel(logging.CRITICAL)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    arguments = build_argument_parser().parse_args()
    orchestrator_client = OrchestratorClient(arguments.orchestrator_url)
    if arguments.command == "stream":
        video_segments = resolve_video_segments(arguments.sample_name_or_video_path)
        session = orchestrator_client.create_session(arguments.user_id)
        logger.info(f"session {session['session_id']}")
        asyncio.run(GlassesVideoStreamer().stream(session["livekit_url"], session["token"], video_segments))
    elif arguments.command == "search":
        search_results = orchestrator_client.search_frames(arguments.user_id, arguments.query, arguments.limit)
        print(json.dumps(search_results, indent=2))
    elif arguments.command == "frame":
        output_path = Path(arguments.output_path or f"{arguments.frame_id}.jpg")
        output_path.write_bytes(orchestrator_client.download_frame_jpeg(arguments.frame_id))
        logger.info(f"saved {output_path}")


if __name__ == "__main__":
    main()
