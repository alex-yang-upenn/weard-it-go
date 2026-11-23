from scene_selection.object_sighting_tracker import DetectedFrame, ObjectSightingTracker, SelectedScene


class SceneSelector:
    def __init__(self, snapshot_interval_seconds: float | None):
        self.snapshot_interval_seconds = snapshot_interval_seconds
        self.tracker_by_session_id: dict[str, ObjectSightingTracker] = {}

    def select_scenes(self, detected_frame: DetectedFrame) -> list[SelectedScene]:
        session_id = detected_frame.sampled_frame.session_id
        if session_id not in self.tracker_by_session_id:
            self.tracker_by_session_id[session_id] = ObjectSightingTracker(self.snapshot_interval_seconds)
        return self.tracker_by_session_id[session_id].select_scenes(detected_frame)

    def end_session(self, session_id: str) -> list[SelectedScene]:
        tracker = self.tracker_by_session_id.pop(session_id, None)
        if tracker is None:
            return []
        return tracker.flush_remaining_left_scenes()
