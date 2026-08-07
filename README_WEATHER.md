# README_WEATHER

This file summarizes the homework-specific design decisions. The full runbook is
in `README.md`.

## Source

I chose the National Weather Service API (`api.weather.gov`) because it is
free, keyless, and returns narrative alert and forecast text that is ideal for
embeddings. The client reads active alerts and gridpoint forecasts after
resolving each location through `/points/{lat},{lon}`.

## Tables

`weather_documents` is the raw/provenance table:

- stable `id`
- `location`, `latitude`, `longitude`, NWS office/grid columns
- `source_type` as `alert` or `forecast`
- `headline`, `event`, `narrative_text`
- timestamps plus raw `payload`

`weather_embeddings` is the retrieval table:

- one row per document chunk
- `embedding vector(384)`
- `model_name`
- HNSW cosine index

## Embeddings

The embedding model is `sentence-transformers/all-MiniLM-L6-v2`, matching the
Day 2 convention and pgvector dimension of 384. Chunking uses `800` characters
with `100` characters of overlap. Most forecasts are short, but alert
description plus instruction text can be long enough to benefit from chunking.

## Run Order

1. Get a Lakebase native-password connection URL.
2. Store it with `python setup_secrets.py` as `database/lakebase-url`.
3. Create tables with `sql/01_setup_weather_tables.sql`, `POST /weather/schema`,
   or by letting the app/notebook call `lakebase.ensure_weather_schema()`.
4. Sync:

   ```bash
   curl -X POST <app-url>/weather/sync \
     -H "Content-Type: application/json" \
     -d '{"locations":["Chicago, IL","Austin, TX"],"limit":50}'
   ```

5. Embed:

   ```bash
   python notebooks/ingest_weather_embeddings.py
   ```

6. Search:

   ```bash
   curl -X POST <app-url>/weather/search \
     -H "Content-Type: application/json" \
     -d '{"query":"flash flood risk this weekend","top_k":5}'
   ```

## Extra Credit

Implemented: upserts/deduplication, source-type filtering, GET search variant,
HNSW vector index, scheduled Databricks Workflow resource, and a complete UI.
