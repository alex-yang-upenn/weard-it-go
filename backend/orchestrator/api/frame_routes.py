import asyncio

from fastapi import APIRouter, HTTPException, Request, Response
from common.raw_frames.raw_frame_codec import convert_i420_to_rgb, decompress_raw_frame, encode_rgb_as_jpeg

frame_router = APIRouter()


def load_stored_frame_as_jpeg(frame_archive, file_path: str, width: int, height: int) -> bytes:
    i420_bytes = decompress_raw_frame(frame_archive.load_frame(file_path))
    return encode_rgb_as_jpeg(convert_i420_to_rgb(i420_bytes, width, height))


@frame_router.get("/frames/{frame_id}")
async def get_frame_as_jpeg(frame_id: str, request: Request) -> Response:
    services = request.app.state
    frame_file = await services.frame_search_repository.find_frame_file(frame_id)
    if frame_file is None:
        raise HTTPException(status_code=404)
    jpeg = await asyncio.to_thread(
        load_stored_frame_as_jpeg, services.frame_archive,
        frame_file["file_path"], frame_file["width"], frame_file["height"],
    )
    return Response(content=jpeg, media_type="image/jpeg")
