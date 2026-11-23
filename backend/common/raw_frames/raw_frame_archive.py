from pathlib import Path


class RawFrameArchive:
    def __init__(self, root_directory: str):
        self.root_directory = Path(root_directory)

    def save_frame(self, session_id: str, frame_index: int, compressed_frame_bytes: bytes) -> str:
        relative_path = f"{session_id}/{frame_index}.i420.zst"
        absolute_path = self.root_directory / relative_path
        absolute_path.parent.mkdir(parents=True, exist_ok=True)
        absolute_path.write_bytes(compressed_frame_bytes)
        return relative_path

    def load_frame(self, relative_path: str) -> bytes:
        return (self.root_directory / relative_path).read_bytes()
