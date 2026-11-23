from pathlib import Path

import httpx

from samples.sample_video_catalog import MixkitClipSegment, SampleVideo
from streaming.video_segment import VideoSegment

BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"


class MixkitClipDownloader:
    def __init__(self, cache_directory: Path):
        self.cache_directory = cache_directory

    def find_or_download_clip(self, clip_segment: MixkitClipSegment) -> Path:
        clip_path = self.cache_directory / f"{clip_segment.mixkit_video_id}-720.mp4"
        if not clip_path.exists():
            self.cache_directory.mkdir(parents=True, exist_ok=True)
            response = httpx.get(
                clip_segment.download_url, headers={"User-Agent": BROWSER_USER_AGENT}, follow_redirects=True, timeout=300
            )
            response.raise_for_status()
            clip_path.write_bytes(response.content)
        return clip_path

    def find_or_download_sample(self, sample: SampleVideo) -> list[VideoSegment]:
        return [
            VideoSegment(self.find_or_download_clip(segment), segment.start_seconds, segment.duration_seconds)
            for segment in sample.segments
        ]
