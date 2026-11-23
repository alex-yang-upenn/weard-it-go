from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoSegment:
    video_path: Path
    start_seconds: float
    duration_seconds: float
