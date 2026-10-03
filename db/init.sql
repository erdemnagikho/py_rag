-- pyrag schema. Loaded automatically by the postgres docker image the first
-- time the container starts on an empty data volume (files in
-- /docker-entrypoint-initdb.d/ are executed in lexical order).
--
-- If you change EMBED_DIM in .env, change the vector(N) dimension below to
-- match, then `docker compose down -v && docker compose up -d` to reset.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    source_path  TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    chunk_count  INT NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    document_id  BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INT NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(768) NOT NULL,
    content_tsv  tsvector GENERATED ALWAYS AS
                 (to_tsvector('english', content)) STORED,
    metadata     JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_content_tsv_gin
    ON chunks USING gin (content_tsv);
