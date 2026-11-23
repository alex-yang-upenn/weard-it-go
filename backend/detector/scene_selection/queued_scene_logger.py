import logging
from datetime import datetime, timezone

from common.redis_streams.sampled_frame_message import SampledFrameMessage
from scene_selection.object_sighting_tracker import SelectedScene

logger = logging.getLogger(__name__)


def describe_gps(sampled_frame: SampledFrameMessage) -> str:
    if not sampled_frame.metadata:
        return "no GPS"
    telemetry = sampled_frame.metadata["telemetry"]
    return f"GPS {telemetry.get('latitude'):.5f}, {telemetry.get('longitude'):.5f}"


def log_queued_scene(selected_scene: SelectedScene) -> None:
    sampled_frame = selected_scene.detected_frame.sampled_frame
    received_at = datetime.fromtimestamp(sampled_frame.received_at_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S UTC")
    detected_objects = ", ".join(
        f"{detection['label']} ({detection['confidence']:.0%})" for detection in selected_scene.objects
    ) or "nothing detected"
    logger.info(
        f"scene queued for embedding: {selected_scene.kind.upper()}: {detected_objects} | "
        f"user {sampled_frame.user_id} | frame {sampled_frame.frame_id} | "
        f"{sampled_frame.width}x{sampled_frame.height} received {received_at} | {describe_gps(sampled_frame)}"
    )
