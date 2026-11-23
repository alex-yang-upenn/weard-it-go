import io

import torch
from PIL import Image
from qwen3_vl_embedding import Qwen3VLEmbedder

QWEN_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
SEARCH_QUERY_INSTRUCTION = "Retrieve the photo that shows the object or place described."
MAX_IMAGE_PIXELS = 640 * 480


class QwenMultimodalEmbedder:
    def __init__(self):
        self.reference_embedder = Qwen3VLEmbedder(
            model_name_or_path=QWEN_EMBEDDING_MODEL_NAME, max_pixels=MAX_IMAGE_PIXELS, torch_dtype=torch.bfloat16
        )

    def embed_jpeg_images(self, jpeg_images: list[bytes]) -> list[list[float]]:
        images = [Image.open(io.BytesIO(jpeg)).convert("RGB") for jpeg in jpeg_images]
        embeddings = self.reference_embedder.process([{"image": image} for image in images])
        return embeddings.float().cpu().tolist()

    def embed_search_text(self, text: str) -> list[float]:
        embeddings = self.reference_embedder.process([{"text": text, "instruction": SEARCH_QUERY_INSTRUCTION}])
        return embeddings[0].float().cpu().tolist()
