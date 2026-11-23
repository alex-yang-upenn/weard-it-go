import httpx


class EmbedderClient:
    def __init__(self, embedder_url: str):
        self.http_client = httpx.AsyncClient(base_url=embedder_url, timeout=600)

    async def embed_search_text(self, text: str) -> list[float]:
        response = await self.http_client.post("/embed/text", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]
