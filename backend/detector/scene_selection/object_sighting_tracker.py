from dataclasses import dataclass

import numpy as np
from common.redis_streams.sampled_frame_message import SampledFrameMessage

SECONDS_UNSEEN_BEFORE_OBJECT_LEFT = 3
SECONDS_UNSEEN_BEFORE_OBJECT_FORGOTTEN = 30
MINIMUM_VIEW_CHANGE_FOR_SNAPSHOT = 0.08


@dataclass
class DetectedFrame:
    sampled_frame: SampledFrameMessage
    rgb: np.ndarray
    detections: dict[str, dict]


@dataclass
class SelectedScene:
    kind: str
    objects: list[dict]
    detected_frame: DetectedFrame


def compute_view_thumbnail(rgb: np.ndarray, thumbnail_size: int = 16) -> np.ndarray:
    height, width = rgb.shape[:2]
    block_height, block_width = max(1, height // thumbnail_size), max(1, width // thumbnail_size)
    cropped = rgb[: block_height * thumbnail_size, : block_width * thumbnail_size].mean(axis=2)
    grayscale = cropped.reshape(thumbnail_size, block_height, thumbnail_size, block_width).mean(axis=(1, 3)) / 255
    return grayscale - grayscale.mean()


def sort_by_confidence(detections: list[dict]) -> list[dict]:
    return sorted(detections, key=lambda detection: -detection["confidence"])


class ObjectSightingTracker:
    """Per-session change detection.

    arrived:  seen in two consecutive frames, not re-announced until forgotten.
    left:     an arrived object unseen for a few seconds; points at the last frame it was in.
    snapshot: at most every snapshot interval, when the view changed enough since the last scene.
    """

    def __init__(self, snapshot_interval_seconds: float | None):
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.labels_seen_last_frame_not_announced: set[str] = set()
        self.last_seen_seconds_by_label: dict[str, float] = {}
        self.announced_labels: set[str] = set()
        self.labels_visible_last_frame: set[str] = set()
        self.last_frame_with_label_in_view: dict[str, DetectedFrame] = {}
        self.last_scene_thumbnail: np.ndarray | None = None
        self.last_scene_seconds = float("-inf")

    def select_scenes(self, detected_frame: DetectedFrame) -> list[SelectedScene]:
        now_seconds = detected_frame.sampled_frame.received_at_ms / 1000
        detections = detected_frame.detections
        visible_labels = set(detections)
        for label in visible_labels:
            self.last_seen_seconds_by_label[label] = now_seconds
        for label in list(self.announced_labels):
            if now_seconds - self.last_seen_seconds_by_label[label] > SECONDS_UNSEEN_BEFORE_OBJECT_FORGOTTEN:
                self.announced_labels.discard(label)

        scenes: list[SelectedScene] = []
        confirmed_labels = (visible_labels & self.labels_seen_last_frame_not_announced) - self.announced_labels
        self.labels_seen_last_frame_not_announced = visible_labels - self.announced_labels
        if confirmed_labels:
            self.announced_labels |= confirmed_labels
            objects = sort_by_confidence([detections[label] for label in confirmed_labels])
            scenes.append(SelectedScene("arrived", objects, detected_frame))

        for label in visible_labels & self.announced_labels:
            if label in self.last_frame_with_label_in_view or label in self.labels_visible_last_frame:
                self.last_frame_with_label_in_view[label] = detected_frame
        self.labels_visible_last_frame = visible_labels

        departed_labels = [
            label for label in self.last_frame_with_label_in_view
            if now_seconds - self.last_seen_seconds_by_label[label] > SECONDS_UNSEEN_BEFORE_OBJECT_LEFT
        ]
        scenes += self.build_left_scenes(departed_labels)

        if not scenes and self.is_snapshot_due(detected_frame, now_seconds):
            scenes.append(SelectedScene("snapshot", sort_by_confidence(list(detections.values())), detected_frame))
        if any(scene.detected_frame is detected_frame for scene in scenes):
            self.last_scene_seconds = now_seconds
            self.last_scene_thumbnail = compute_view_thumbnail(detected_frame.rgb)
        return scenes

    def is_snapshot_due(self, detected_frame: DetectedFrame, now_seconds: float) -> bool:
        if self.snapshot_interval_seconds is None:
            return False
        if now_seconds - self.last_scene_seconds < self.snapshot_interval_seconds:
            return False
        thumbnail = compute_view_thumbnail(detected_frame.rgb)
        if self.last_scene_thumbnail is None or self.last_scene_thumbnail.shape != thumbnail.shape:
            return True
        return float(np.abs(thumbnail - self.last_scene_thumbnail).mean()) >= MINIMUM_VIEW_CHANGE_FOR_SNAPSHOT

    def flush_remaining_left_scenes(self) -> list[SelectedScene]:
        return self.build_left_scenes(list(self.last_frame_with_label_in_view))

    def build_left_scenes(self, departed_labels: list[str]) -> list[SelectedScene]:
        labels_by_frame_id: dict[str, tuple[DetectedFrame, list[str]]] = {}
        for label in departed_labels:
            last_frame = self.last_frame_with_label_in_view.pop(label)
            labels_by_frame_id.setdefault(last_frame.sampled_frame.frame_id, (last_frame, []))[1].append(label)
        return [
            SelectedScene("left", sort_by_confidence([last_frame.detections[label] for label in labels]), last_frame)
            for last_frame, labels in labels_by_frame_id.values()
        ]
