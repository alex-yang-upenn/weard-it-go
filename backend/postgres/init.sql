CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ
);

CREATE TABLE frames (
    frame_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    frame_index BIGINT NOT NULL,
    received_at_ms BIGINT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    file_path TEXT NOT NULL,
    metadata JSONB
);

CREATE TABLE telemetry (
    session_id TEXT NOT NULL,
    received_at_ms BIGINT NOT NULL,
    payload JSONB NOT NULL
);

CREATE TABLE scenes (
    scene_id TEXT PRIMARY KEY,
    frame_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    objects JSONB NOT NULL,
    created_at_ms BIGINT NOT NULL
);

CREATE TABLE embeddings (
    frame_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    embedding VECTOR(2048) NOT NULL
);
