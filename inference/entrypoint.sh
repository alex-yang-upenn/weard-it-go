#!/usr/bin/env bash
# Starts the two vLLM servers (embedder first, then the VLM: starting them at the
# same time confuses vLLM's GPU memory profiling), waits until each is healthy,
# then runs the queue worker. If any of the three exits, the container exits so
# Docker restarts it.
set -euo pipefail

: "${VLM_MODEL:=Qwen/Qwen3-VL-8B-Instruct-FP8}"
: "${EMBED_MODEL:=Qwen/Qwen3-Embedding-0.6B}"
: "${VLM_GPU_UTIL:=0.70}"     # fractions of total GPU memory; together must stay < ~0.9
: "${EMBED_GPU_UTIL:=0.15}"
: "${VLM_MAX_LEN:=8192}"
: "${VLM_EXTRA_ARGS:=}"       # passed through to `vllm serve`, e.g. "--enforce-eager"
: "${EMBED_EXTRA_ARGS:=}"
: "${STARTUP_TIMEOUT_S:=1800}" # first run downloads the weights
export VLM_MODEL EMBED_MODEL

pids=()
cleanup() { kill "${pids[@]}" 2>/dev/null || true; wait || true; }
trap cleanup EXIT
trap 'exit 143' TERM INT

wait_healthy() {  # name port pid
  local deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  until python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:$2/health', timeout=2)" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || { echo "[entrypoint] $1 exited during startup" >&2; exit 1; }
    ((SECONDS < deadline)) || { echo "[entrypoint] $1 not healthy after ${STARTUP_TIMEOUT_S}s" >&2; exit 1; }
    sleep 3
  done
  echo "[entrypoint] $1 ready on :$2"
}

echo "[entrypoint] starting embedder $EMBED_MODEL"
# shellcheck disable=SC2086
vllm serve "$EMBED_MODEL" --host 127.0.0.1 --port 8001 \
  --runner pooling --gpu-memory-utilization "$EMBED_GPU_UTIL" --max-model-len 2048 \
  $EMBED_EXTRA_ARGS &
pids+=($!)
wait_healthy embedder 8001 "${pids[-1]}"

echo "[entrypoint] starting VLM $VLM_MODEL"
# shellcheck disable=SC2086
vllm serve "$VLM_MODEL" --host 127.0.0.1 --port 8000 \
  --gpu-memory-utilization "$VLM_GPU_UTIL" --max-model-len "$VLM_MAX_LEN" \
  --limit-mm-per-prompt '{"image": 1}' \
  $VLM_EXTRA_ARGS &
pids+=($!)
wait_healthy vlm 8000 "${pids[-1]}"

python3 /app/worker.py &
pids+=($!)

# Exit as soon as any child does; the EXIT trap stops the rest.
set +e
wait -n
status=$?
echo "[entrypoint] a process exited (status $status), shutting down" >&2
exit "$status"
