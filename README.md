# Wear'd It Go?

Smart-glasses video goes in, and you can ask where you left your things. `ios/` is the phone app, `backend/` holds the services, and `test_iphone/` stands in for the phone.

## Backend

Each folder in `backend/` is one Docker service, wired together in `backend/docker-compose.yml`. Video flows phone → LiveKit → ingest → detector → embedder, and the orchestrator answers searches over what the embedder indexed.

**orchestrator** is the only service the phone talks to over HTTP, on port 8000. `POST /sessions` records a session in Postgres, creates a LiveKit room tagged with the user, and hands back the room URL and a join token. `GET /search` asks the embedder to embed the query text, then runs a pgvector search over that user's frames. `GET /frames/{frame_id}` reads a stored raw frame from disk and returns it as a JPEG.

**livekit** is a stock LiveKit server configured by `livekit/livekit.yaml`. The phone joins the room the orchestrator made, publishes its camera as a video track, and sends GPS as JSON on the data topic `telemetry`. `LIVEKIT_NODE_IP` must be your machine's LAN IP so the phone can reach the media ports (7880, 7881 and UDP 50000-50100).

**ingest** is a LiveKit agent that joins every room and records until the phone leaves. It stores every frame, zstd-compressed I420, under `ingest/data/frames/<session_id>/`, with a row in the Postgres `frames` table that carries the latest GPS reading. Telemetry also goes to the `telemetry` table. It samples 8 frames per second onto the Redis `sampled_frames` stream, and when the phone leaves it marks the session ended and adds a `session_ended` message to that stream.

**detector** reads `sampled_frames` in batches and runs YOLOE, prompted with a fixed list of commonly misplaced objects (keys, wallet, phone, glasses and so on). A per-session tracker turns the detections into scenes: `arrived` when an object shows up in two frames in a row, `left` when an object has not been seen for 3 seconds (pointing at the last frame it was in), and an optional `snapshot` when the view has changed enough. Each scene goes onto the Redis `scenes` stream with its JPEG and detected objects.

**embedder** reads `scenes`, writes each one to the Postgres `scenes` table, and embeds each scene's image with Qwen3-VL-Embedding-2B into the `embeddings` table. It also serves `POST /embed/text` on port 8001, which the orchestrator uses to embed search queries into the same vector space. Only frames that became scenes are searchable.

**redis** carries two streams, each read by a consumer group: `sampled_frames` (ingest → detector) and `scenes` (detector → embedder). Both are capped at about 1000 messages, and unacknowledged messages are re-read when a consumer restarts.

**postgres** runs pgvector, with the schema in `postgres/init.sql`. The tables are `sessions`, `frames` (every stored frame and where it is on disk), `telemetry`, `scenes` and `embeddings` (one 2048-dimension vector per indexed frame). Search orders a user's embeddings by cosine distance (`<=>`) to the query embedding and joins `frames` for the session, time and GPS metadata.

**common** holds the code the services share: the Postgres connection pool, the raw-frame archive and codec (zstd plus I420 to RGB and JPEG), and the Redis stream consumer and message formats for both streams.


## Usage

### Run the backend

1. Install Docker Desktop and give it at least 8 GB of memory.
2. In `backend/`, copy `.env.example` to `.env` and set `LIVEKIT_NODE_IP` to your machine's LAN IP.
3. From `backend/`, run `docker compose up -d --build`.

Set `SNAPSHOT_INTERVAL_S` in `.env` to also index a `snapshot` scene at most every N seconds when the view has changed. It is off by default.


### test_iphone

```
cd test_iphone
python -m venv .venv                        # then activate it: .venv\Scripts\activate (Windows) or source .venv/bin/activate
pip install -r requirements.txt
python main.py stream phone_on_laptop        # start a session, stream a sample (or any .mp4 path) with fake GPS
python main.py search "where is my phone"    # print search results as JSON (--limit, default 5)
python main.py frame <frame_id> [out.jpg]    # save a stored frame (default <frame_id>.jpg)
```

`--orchestrator-url` (default `http://localhost:8000`) and `--user-id` (default `test-iphone`) go before the command, e.g. `python main.py --user-id alice search "keys"`.

Each sample is a few [Mixkit](https://mixkit.co/) clips under the [Mixkit Stock Video Free License](https://mixkit.co/license/#videoFree), played back to back: an object, then a cut to somewhere else. Clips and trim ranges are in `samples/sample_video_catalog.py`, and each clip is downloaded into `sample_videos/` the first time it's used.

- `phone_on_laptop`: a phone on a laptop, then a living room
- `watch_at_keyboard`: hands with a watch typing, then a Venice alley
- `laptop_by_the_window`: a woman on a laptop, then a room interior
- `first_person_laptop_with_coffee`: a first-person laptop and coffee, then a house corridor

Results are checked ad hoc, for example by looking at the backend logs or querying Postgres:

```
cd backend
docker compose exec postgres psql -U weardit
```

## TODOs and extensions

iOS app:
- Build and test the new backend wiring on a real device. It hasn't been compiled since the change. The app calls `POST /sessions` with a per-install `user_id` (shown on the Stream screen) and sends GPS on the LiveKit `telemetry` data topic while streaming. Set the backend URL in Settings to `http://<LAN IP>:8000`.

Backend:
- Tune the search cutoff (`max_cosine_distance`, default 0.9) on real footage. Good matches have scored about 0.36–0.7.
- Store frames in S3 instead of `backend/ingest/data/frames`.
- Deploy to AWS with infra-as-code. Use CUDA torch wheels instead of the CPU index in the detector and embedder Dockerfiles, and give those containers an NVIDIA GPU.
- Auth: right now the phone just sends a `user_id` string.
- Phone alerts when an object arrives or leaves.
- A bigger embedding model than Qwen3-VL-Embedding-2B.
