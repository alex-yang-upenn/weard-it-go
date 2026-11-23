class LatestTelemetryTracker:
    def __init__(self):
        self.latest_telemetry: dict | None = None
        self.latest_telemetry_received_at_ms: int | None = None

    def record(self, telemetry: dict, received_at_ms: int) -> None:
        self.latest_telemetry = telemetry
        self.latest_telemetry_received_at_ms = received_at_ms

    def build_frame_metadata(self) -> dict | None:
        if self.latest_telemetry is None:
            return None
        return {"telemetry": self.latest_telemetry, "telemetry_received_at_ms": self.latest_telemetry_received_at_ms}
