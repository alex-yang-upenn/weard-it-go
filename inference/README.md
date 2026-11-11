# Inference server

Turns saved scenes (frame + YOLOE detections) into searchable memories in turbopuffer.

`live_agent.py` saves three kinds of scene per video stream:

| kind | when | why |
|---|---|---|
| `arrived` | an object comes into view (also alerts the phone) | first sighting |
| `left` | an object has been unseen for `LEFT_VIEW_S` (3 s); saved with the **last frame it was in** | where it was put down |
| `snapshot` | at most every `SNAPSHOT_EVERY_S` (20 s) when the view changed | catches what YOLOE misses; the VLM lists personal items it sees |

```
live_agent.py ──▶ Redis stream "scenes:index" ──▶ inference container(s) ──▶ turbopuffer "scenes"
                                                  ├ vLLM VLM: caption + where each object is
                                                  └ vLLM embedder: vector of that text
```

One container = one GPU running two vLLM models and `worker.py`. Every container joins the
same Redis consumer group, so Redis hands each scene to exactly one of them. To scale,
start more containers (one per GPU) pointed at the same Redis.

## Run (GPU box with the NVIDIA container toolkit)

```sh
cd inference
cp .env.example .env        # set TURBOPUFFER_API_KEY, TURBOPUFFER_REGION, REDIS_PASSWORD
docker compose up --build   # starts Redis + one inference server
```

First start downloads ~11 GB of weights into the `hf-cache` volume. Watch for
`vlm ready` and then `consumer ... concurrency 8`.

Point the backend at this Redis (on the Mac):

```sh
cd backend && .venv/bin/pip install -r requirements.txt
REDIS_URL=redis://:<password>@<gpu-box>:6379 ./dev.sh
```

`REDIS_URL` unset = the backend works as before and queues nothing.

Backfill scenes already on disk (also the quickest test, no glasses needed):

```sh
REDIS_URL=redis://:<password>@<gpu-box>:6379 python inference/enqueue_scenes.py
```

## More servers

- **Another GPU machine:** copy this folder, set `REDIS_URL=redis://:<password>@<redis-host>:6379`
  in `.env`, run `docker compose up --build inference`.
- **More GPUs on one machine:** add a second service like `inference` with `device_ids: ["1"]`.
  Don't put two containers on the same GPU; each one reserves ~85% of its memory.

## Queue behavior

- A job is acked only after its row is written to turbopuffer.
- Workers heartbeat the jobs they're running. Jobs from a worker that died are taken over
  by another worker after `CLAIM_IDLE_MS` (2 min).
- Failed jobs retry; after `MAX_ATTEMPTS` (3) they move to `scenes:index:dead` with a reason.
- Rows are keyed by the scene id (same as the scene file name), so re-processing just overwrites.

Useful checks:

```sh
redis-cli -a <password> XLEN scenes:index
redis-cli -a <password> XPENDING scenes:index indexers
redis-cli -a <password> XRANGE scenes:index:dead - +
docker compose exec inference python3 -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8080/healthz').read().decode())"
```

## turbopuffer row

| field | |
|---|---|
| `id` | `<ts_ms>_<identity>_<kind>` (older scenes: `<ts_ms>_<identity>`) |
| `kind` | `arrived` / `left` / `snapshot`; for "where is X", prefer the newest `left` row |
| `vector` | embedding of `text` (cosine) |
| `text` | caption + place + per-object placement (full-text indexed) |
| `caption`, `place` | from the VLM |
| `labels` | detector labels ∪ VLM labels, for exact filters |
| `objects_json` | detector boxes + VLM placements |
| `ts_ms`, `identity`, `image` | from the scene |
| `lat`, `lon`, `location_accuracy_m` | when the phone sent a location |

Queries must embed the question with the same `EMBED_MODEL`. Qwen3-Embedding works best
with an instruction prefix on queries, e.g.
`"Instruct: Find where this object was last seen\nQuery: where are my keys"`.

## Models

Defaults fit one 24 GB GPU (L4 / A10 / 4090). Change them in `.env`:

| var | default |
|---|---|
| `VLM_MODEL` | `Qwen/Qwen3-VL-8B-Instruct-FP8` |
| `EMBED_MODEL` | `Qwen/Qwen3-Embedding-0.6B` |
| `VLM_GPU_UTIL` / `EMBED_GPU_UTIL` | `0.70` / `0.15` |

Changing `EMBED_MODEL` changes vector size: use a new `TPUF_NAMESPACE` and re-run the backfill.
