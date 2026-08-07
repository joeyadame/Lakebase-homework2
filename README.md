# Weather Intelligence Lakebase App

Weather Intelligence is a Databricks App that harvests public National Weather
Service text, stores it in Lakebase, embeds it with pgvector, and exposes
semantic search over weather alerts and forecasts.

Project structure:

- `app.py` is the Flask app with `/weather/sync`, `/weather/embed`,
  `/weather/search`, and the UI.
- `lakebase.py` owns the single `LAKEBASE_URL` psycopg2 connection helper.
- `weather_client.py` harvests and normalizes NWS alerts and forecasts.
- `/weather/embed` computes chunk embeddings from the deployed app environment,
  which uses the same Lakebase network path that already works for sync.
- `sql/01_setup_weather_tables.sql` contains the explicit empty-database DDL.

## Data Source

The app uses `api.weather.gov` because it is free, keyless, and provides useful
unstructured text:

- active alerts: `description`, `instruction`, `headline`, `event`
- grid forecasts: `detailedForecast`, `shortForecast`, wind, temperature, PoP

NWS requires a descriptive `User-Agent`, so set `NWS_USER_AGENT` in `app.yaml`
before deploying. The weather records themselves come from NWS. For convenience,
`weather_client.py` accepts either `lat,lon` or city/state labels. City labels
use a small built-in demo map first, then public geocoding fallback, because the
NWS `/points/{lat},{lon}` endpoint requires coordinates.

## Lakebase Setup

Start with an empty Lakebase database.

1. In Databricks, create a Lakebase instance.
2. Enable native password authentication.
3. Create a native-password role for the app.
4. Copy the role's full Postgres connection URL:

   ```text
   postgresql://<role>:<password>@<host>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require
   ```

5. Store it in the Databricks secret expected by `app.yaml`:

   ```bash
   python setup_secrets.py
   ```

   The script stores `database/lakebase-url`. If your copied URL omits the
   password, the script prompts for it and builds the complete URL.

6. Create the tables one of three ways:

   ```sql
   -- Run in Lakebase SQL editor or psql
   \i sql/01_setup_weather_tables.sql
   ```

   or deploy the app and click `Init Schema`, or run:

   ```bash
   curl -X POST https://<app-url>/weather/schema
   ```

## Schema

`weather_documents` stores raw normalized weather text:

- `id` stable NWS-derived primary key
- `location`, `latitude`, `longitude`, `office`, `grid_x`, `grid_y`
- `source_type` as `alert` or `forecast`
- `headline`, `event`, `narrative_text`
- `issued_at`, `effective_at`, `expires_at`
- `payload` raw JSON provenance and `synced_at`

`weather_embeddings` stores chunk-level vectors:

- `id`, `document_id`, `chunk_index`, `chunk_text`
- `embedding vector(384)`
- `model_name`, `created_at`
- HNSW cosine index: `USING hnsw (embedding vector_cosine_ops)`

The embedding model is `sentence-transformers/all-MiniLM-L6-v2` because it is
the assignment-specified model and emits 384-dimensional vectors. Chunking uses
`CHUNK_SIZE=800` and `CHUNK_OVERLAP=100`; most NWS records are short, but alerts
with instructions can still benefit from stable chunking.

## Local Run

```bash
cp .env.example .env
# edit .env and paste LAKEBASE_URL
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`.

## End-To-End Pipeline

1. Sync raw NWS documents:

   ```bash
   curl -X POST http://localhost:8000/weather/sync \
     -H "Content-Type: application/json" \
     -d '{"locations":["Chicago, IL","Austin, TX"],"limit":50}'
   ```

2. Embed new, unembedded rows from the deployed app:

   ```bash
   curl -X POST http://localhost:8000/weather/embed \
     -H "Content-Type: application/json" \
     -d '{"limit":25}'
   ```

   This only selects documents with no existing rows in `weather_embeddings`.
   Re-running it is safe; already embedded documents are skipped.

3. Search:

   ```bash
   curl -X POST http://localhost:8000/weather/search \
     -H "Content-Type: application/json" \
     -d '{"query":"flash flood risk this weekend","top_k":5}'
   ```

   GET variant:

   ```bash
   curl "http://localhost:8000/weather/search?query=river%20flooding&top_k=5&source_type=alert"
   ```

## Databricks App Deployment

1. Push this repo to GitHub.
2. In Databricks, create or update a Git folder for the repo.
3. Edit `app.yaml` and replace the placeholder email in `NWS_USER_AGENT`.
4. Create a Databricks App from the Git folder root.
5. Deploy. `app.yaml` runs `python app.py` and reads `database/lakebase-url`.
6. Open the app, click `Init Schema`, then `Sync Weather`.
7. Click `Embed New`.
8. Search from the UI or call `/weather/search`.

## API

- `GET /healthz`
- `POST /weather/schema`
- `GET /weather/status`
- `GET /weather/documents?limit=20&source_type=alert`
- `POST /weather/sync`
- `POST /weather/embed`
- `POST /weather/search`
- `GET /weather/search?query=...&top_k=5&source_type=forecast`

## Stretch Goals Included

- Upserts on `weather_documents.id` and `(document_id, chunk_index)`.
- GET search variant.
- Retrieval filter by `source_type`.
- HNSW pgvector cosine index.
- App-side embedding for only-new documents.
- Frontend for sync, embedding, status, recent documents, and vector search.

## Known Limitations

- NWS covers the United States and territories.
- City/state geocoding is convenience plumbing; `lat,lon` is the most reliable
  input for exact locations.
- First embed/search in a fresh app process may take time while the embedding model
  loads into memory.
- The app returns retrieved documents, not an LLM-generated RAG summary. That
  would require adding a model-serving or OpenAI API credential.
