FROM python:3.12-slim
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
ADD https://raw.githubusercontent.com/QwenLM/Qwen3-VL-Embedding/393e2978d27852b0d0230d6994f37f9c15bed73c/src/models/qwen3_vl_embedding.py /qwen_reference/qwen3_vl_embedding.py
ENV PYTHONPATH=/qwen_reference
WORKDIR /app
COPY embedder/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common common
COPY embedder .
CMD ["uvicorn", "main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001", "--log-level", "warning"]
