from fastapi import APIRouter, Request

search_router = APIRouter()


@search_router.get("/search")
async def search_frames(
    request: Request, user_id: str, query: str, limit: int = 10, max_cosine_distance: float = 0.9
) -> list[dict]:
    services = request.app.state
    query_embedding = await services.embedder_client.embed_search_text(query)
    matching_frames = await services.frame_search_repository.find_frames_closest_to_embedding(
        user_id, query_embedding, max_cosine_distance, limit
    )
    return [{**frame, "image_url": f"/frames/{frame['frame_id']}"} for frame in matching_frames]
