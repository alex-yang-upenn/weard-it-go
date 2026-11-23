import json
from dataclasses import dataclass

SCENES_STREAM = "scenes"


@dataclass
class SceneMessage:
    scene_id: str
    frame_id: str
    session_id: str
    user_id: str
    kind: str
    objects: list[dict]
    created_at_ms: int
    jpeg_bytes: bytes

    def to_stream_fields(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "kind": self.kind,
            "objects": json.dumps(self.objects),
            "created_at_ms": self.created_at_ms,
            "jpeg_bytes": self.jpeg_bytes,
        }

    @classmethod
    def from_stream_fields(cls, fields: dict) -> "SceneMessage":
        return cls(
            scene_id=fields[b"scene_id"].decode(),
            frame_id=fields[b"frame_id"].decode(),
            session_id=fields[b"session_id"].decode(),
            user_id=fields[b"user_id"].decode(),
            kind=fields[b"kind"].decode(),
            objects=json.loads(fields[b"objects"]),
            created_at_ms=int(fields[b"created_at_ms"]),
            jpeg_bytes=fields[b"jpeg_bytes"],
        )
