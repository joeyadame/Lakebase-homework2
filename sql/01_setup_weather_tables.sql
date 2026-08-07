-- Weather Intelligence Lakebase schema
-- Run this in the Lakebase SQL editor or through psql before deployment if you
-- want to create the tables explicitly instead of letting the Flask app do it.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_documents (
    id TEXT PRIMARY KEY,
    location TEXT NOT NULL,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    office TEXT,
    grid_x INT,
    grid_y INT,
    source_type TEXT NOT NULL
        CHECK (source_type IN ('alert', 'forecast')),
    headline TEXT,
    event TEXT,
    narrative_text TEXT NOT NULL
        CHECK (length(trim(narrative_text)) > 0),
    issued_at TIMESTAMPTZ,
    effective_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    payload JSONB NOT NULL,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_effective
    ON weather_documents (source_type, effective_at DESC);

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL
        REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL CHECK (length(trim(chunk_text)) > 0),
    embedding VECTOR(384) NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_embedding_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);

SELECT
    table_name,
    column_name,
    data_type,
    udt_name
FROM information_schema.columns
WHERE table_name IN ('weather_documents', 'weather_embeddings')
ORDER BY table_name, ordinal_position;
