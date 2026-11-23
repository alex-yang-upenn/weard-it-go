import httpx


class OrchestratorClient:
    def __init__(self, orchestrator_url: str):
        self.http_client = httpx.Client(base_url=orchestrator_url, timeout=120)

    def create_session(self, user_id: str) -> dict:
        response = self.http_client.post("/sessions", json={"user_id": user_id})
        response.raise_for_status()
        return response.json()

    def search_frames(self, user_id: str, query: str, limit: int) -> list[dict]:
        response = self.http_client.get("/search", params={"user_id": user_id, "query": query, "limit": limit})
        response.raise_for_status()
        return response.json()

    def download_frame_jpeg(self, frame_id: str) -> bytes:
        response = self.http_client.get(f"/frames/{frame_id}")
        response.raise_for_status()
        return response.content
