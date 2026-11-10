# One indexing server: two vLLM models + the Redis queue worker, on one GPU.
# Run one container per GPU; they all share the same Redis queue.
FROM vllm/vllm-openai:v0.11.0

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY entrypoint.sh worker.py prompts.py ./
RUN chmod +x entrypoint.sh

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Worker health endpoint (models + queue loop). Models can take a long time to
# download on first start, hence the long start period.
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=4)"

# The base image's entrypoint is `vllm serve`; we start everything ourselves.
ENTRYPOINT ["/app/entrypoint.sh"]
