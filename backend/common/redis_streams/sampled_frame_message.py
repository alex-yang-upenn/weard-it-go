import json
from dataclasses import dataclass

SAMPLED_FRAMES_STREAM = "sampled_frames"


@dataclass
class SampledFrameMessage:
    frame_id: str
    session_id: str
    user_id: str
    received_at_ms: int
    width: int
    height: int
    compressed_i420_bytes: bytes
    metadata: dict | None

    def to_stream_fields(self) -> dict:
        return {
            "type": "frame",
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "received_at_ms": self.received_at_ms,
            "width": self.width,
            "height": self.height,
            "compressed_i420_bytes": self.compressed_i420_bytes,
            "metadata": json.dumps(self.metadata),
        }

    @classmethod
    def from_stream_fields(cls, fields: dict) -> "SampledFrameMessage":
        return cls(
            frame_id=fields[b"frame_id"].decode(),
            session_id=fields[b"session_id"].decode(),
            user_id=fields[b"user_id"].decode(),
            received_at_ms=int(fields[b"received_at_ms"]),
            width=int(fields[b"width"]),
            height=int(fields[b"height"]),
            compressed_i420_bytes=fields[b"compressed_i420_bytes"],
            metadata=json.loads(fields[b"metadata"]),
        )


def build_session_ended_fields(session_id: str) -> dict:
    return {"type": "session_ended", "session_id": session_id}


def is_session_ended_message(fields: dict) -> bool:
    return fields[b"type"] == b"session_ended"
