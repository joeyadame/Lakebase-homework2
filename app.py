"""
Weather Intelligence Databricks App.

This app follows the Weather Intelligence project pattern:
- Flask serves a small UI and REST API.
- lakebase.py owns the psycopg2 connection to Lakebase.
- weather_client.py harvests public NWS weather text.
- the app embeds new weather documents directly from the environment that
  already has Lakebase access.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from flask import Flask, jsonify, render_template, request
from psycopg2.extras import Json, execute_values
from werkzeug.exceptions import HTTPException

import lakebase
from weather_client import WeatherClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-intelligence")

app = Flask(__name__)

WEATHER_DOCUMENTS_TABLE = os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
WEATHER_EMBEDDINGS_TABLE = os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
EMBEDDING_MODEL_NAME = os.environ.get(
    "WEATHER_EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
EMBEDDING_DIM = int(os.environ.get("WEATHER_EMBEDDING_DIM", "384"))
CHUNK_SIZE = int(os.environ.get("WEATHER_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.environ.get("WEATHER_CHUNK_OVERLAP", "100"))
EMBED_BATCH_SIZE = int(os.environ.get("WEATHER_EMBED_BATCH_SIZE", "32"))
MAX_SYNC_LIMIT = 200
MAX_EMBED_LIMIT = 100
MAX_TOP_K = 20

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("WEATHER_CHUNK_OVERLAP must be smaller than WEATHER_CHUNK_SIZE")

DEFAULT_LOCATIONS = [
    value.strip()
    for value in os.environ.get(
        "WEATHER_DEFAULT_LOCATIONS",
        "Chicago, IL;Austin, TX",
    ).split(";")
    if value.strip()
]

_SCHEMA_READY = False
_EMBEDDING_MODEL: Any | None = None


@app.errorhandler(Exception)
def handle_exception(err):
    if isinstance(err, HTTPException) and err.code and err.code < 500:
        return jsonify({"error": err.description}), err.code
    logger.exception("Unhandled exception while processing request")
    status_code = getattr(err, "code", 500)
    if not isinstance(status_code, int):
        status_code = 500
    return jsonify({"error": str(err)}), status_code


@app.route("/")
def index():
    return render_template(
        "index.html",
        default_locations="\n".join(DEFAULT_LOCATIONS),
        default_top_k=5,
        embedding_model=EMBEDDING_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/weather/schema", methods=["POST"])
def initialize_weather_schema():
    ensure_weather_schema()
    return jsonify(
        {
            "status": "ready",
            "documents_table": WEATHER_DOCUMENTS_TABLE,
            "embeddings_table": WEATHER_EMBEDDINGS_TABLE,
            "embedding_dim": EMBEDDING_DIM,
        }
    )


@app.route("/weather/status", methods=["GET"])
def weather_status():
    ensure_weather_schema()
    rows = lakebase.run_query(
        f"""
        SELECT
            (SELECT COUNT(*)::int FROM {WEATHER_DOCUMENTS_TABLE}) AS documents,
            (SELECT COUNT(*)::int FROM {WEATHER_EMBEDDINGS_TABLE}) AS embeddings,
            (
                SELECT COUNT(*)::int
                FROM {WEATHER_DOCUMENTS_TABLE} d
                WHERE length(trim(COALESCE(d.narrative_text, ''))) > 0
                  AND NOT EXISTS (
                      SELECT 1
                      FROM {WEATHER_EMBEDDINGS_TABLE} e
                      WHERE e.document_id = d.id
                  )
            ) AS unembedded_documents,
            (
                SELECT COALESCE(jsonb_object_agg(source_type, total), '{{}}'::jsonb)
                FROM (
                    SELECT source_type, COUNT(*)::int AS total
                    FROM {WEATHER_DOCUMENTS_TABLE}
                    GROUP BY source_type
                ) source_counts
            ) AS by_source
        """
    )
    row = _serialize_row(rows[0]) if rows else {}
    row.update(
        {
            "documents_table": WEATHER_DOCUMENTS_TABLE,
            "embeddings_table": WEATHER_EMBEDDINGS_TABLE,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embed_batch_size": EMBED_BATCH_SIZE,
        }
    )
    return jsonify({"status": row})


@app.route("/weather/documents", methods=["GET"])
def weather_documents():
    ensure_weather_schema()
    limit = _clamp_int(request.args.get("limit"), default=20, lower=1, upper=100)
    try:
        source_type = _source_type_filter(request.args.get("source_type"))
    except ValueError as exc:
        return _error(str(exc))

    where_sql = ""
    params: list[Any] = []
    if source_type:
        where_sql = "WHERE source_type = %s"
        params.append(source_type)
    params.append(limit)

    rows = lakebase.run_query(
        f"""
        SELECT
            id,
            location,
            source_type,
            headline,
            event,
            narrative_text,
            issued_at,
            effective_at,
            expires_at,
            synced_at
        FROM {WEATHER_DOCUMENTS_TABLE}
        {where_sql}
        ORDER BY synced_at DESC, effective_at DESC NULLS LAST
        LIMIT %s
        """,
        tuple(params),
    )
    return jsonify({"documents": _serialize_rows(rows)})


@app.route("/weather/sync", methods=["POST"])
def sync_weather():
    """Harvest NWS alerts/forecasts and upsert raw documents into Lakebase."""
    ensure_weather_schema()
    body = _payload()
    locations = _normalize_locations(body.get("locations")) or DEFAULT_LOCATIONS
    if not locations:
        return _error("At least one location is required.")

    limit = _clamp_int(body.get("limit"), default=50, lower=1, upper=MAX_SYNC_LIMIT)
    include_hourly = bool(body.get("include_hourly", False))

    client = WeatherClient()
    documents, errors = client.fetch_documents_for_locations(
        locations,
        limit=limit,
        include_hourly=include_hourly,
    )
    synced = _upsert_weather_documents(documents)
    source_counts = Counter(doc["source_type"] for doc in documents)

    status_code = 207 if errors and documents else 200
    if errors and not documents:
        status_code = 400

    return (
        jsonify(
            {
                "synced": synced,
                "requested_locations": locations,
                "source_counts": dict(source_counts),
                "errors": errors,
                "next_step": "POST /weather/embed to embed new documents, then POST /weather/search.",
            }
        ),
        status_code,
    )


@app.route("/weather/embed", methods=["POST"])
def embed_weather():
    """Embed newly synced weather documents that have no embeddings yet."""
    ensure_weather_schema()
    body = _payload()
    limit = _clamp_int(body.get("limit"), default=25, lower=1, upper=MAX_EMBED_LIMIT)
    try:
        source_type = _source_type_filter(body.get("source_type"))
    except ValueError as exc:
        return _error(str(exc))

    result = _embed_unembedded_weather_documents(limit=limit, source_type=source_type)
    return jsonify(
        {
            "documents_selected": result["documents_selected"],
            "chunks_prepared": result["chunks_prepared"],
            "embeddings_inserted": result["embeddings_inserted"],
            "model_name": EMBEDDING_MODEL_NAME,
            "embedding_dim": EMBEDDING_DIM,
            "source_type": source_type or "all",
            "message": (
                "No new weather documents need embeddings."
                if result["documents_selected"] == 0
                else "Embedded newly synced weather documents."
            ),
        }
    )


@app.route("/weather/search", methods=["GET", "POST"])
def search_weather():
    ensure_weather_schema()
    data = _search_payload()
    query = _clean_text(data.get("query"), collapse=True)
    if not query:
        return _error("Search query is required.")

    top_k = _clamp_int(data.get("top_k"), default=5, lower=1, upper=MAX_TOP_K)
    try:
        source_type = _source_type_filter(data.get("source_type"))
    except ValueError as exc:
        return _error(str(exc))

    embedding_count = lakebase.run_query(
        f"SELECT COUNT(*)::int AS total FROM {WEATHER_EMBEDDINGS_TABLE}"
    )[0]["total"]
    if embedding_count == 0:
        return jsonify(
            {
                "query": query,
                "matches": [],
                "message": "No weather embeddings found. Run /weather/sync, then POST /weather/embed.",
            }
        )

    query_vector = _embed_query(query)
    where_sql = ""
    params: list[Any] = [query_vector]
    if source_type:
        where_sql = "WHERE d.source_type = %s"
        params.append(source_type)
    params.extend([query_vector, top_k])

    rows = lakebase.run_query(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.event,
            d.narrative_text,
            d.issued_at,
            d.effective_at,
            e.id AS embedding_id,
            e.chunk_index,
            e.chunk_text,
            e.model_name,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM {WEATHER_EMBEDDINGS_TABLE} e
        JOIN {WEATHER_DOCUMENTS_TABLE} d
            ON d.id = e.document_id
        {where_sql}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )

    matches = _serialize_rows(rows)
    return jsonify(
        {
            "query": query,
            "top_k": top_k,
            "source_type": source_type or "all",
            "matches": matches,
        }
    )


def ensure_weather_schema() -> None:
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    lakebase.ensure_weather_schema(
        documents_table=WEATHER_DOCUMENTS_TABLE,
        embeddings_table=WEATHER_EMBEDDINGS_TABLE,
        embedding_dim=EMBEDDING_DIM,
    )
    _SCHEMA_READY = True


def _payload() -> dict[str, Any]:
    return (request.get_json(silent=True) or {}) if request.is_json else {}


def _search_payload() -> dict[str, Any]:
    if request.method == "GET":
        return {
            "query": request.args.get("query"),
            "top_k": request.args.get("top_k"),
            "source_type": request.args.get("source_type"),
        }
    return _payload()


def _normalize_locations(value: Any) -> list[str | dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            item
            for item in value
            if (isinstance(item, dict) or (isinstance(item, str) and item.strip()))
        ]
    if isinstance(value, str):
        raw_parts = value.replace("\r", "\n").replace(";", "\n").split("\n")
        return [part.strip() for part in raw_parts if part.strip()]
    return []


def _source_type_filter(value: Any) -> str | None:
    source_type = _clean_text(value, collapse=True).lower()
    if not source_type or source_type == "all":
        return None
    if source_type not in {"alert", "forecast"}:
        raise ValueError("source_type must be one of: all, alert, forecast")
    return source_type


def _clean_text(value: Any, *, collapse: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if collapse:
        value = " ".join(value.split())
    return value


def _clamp_int(value: Any, *, default: int, lower: int, upper: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, lower), upper)


def _error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _json_ready(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _serialize_row(row: Any) -> dict[str, Any]:
    return {key: _json_ready(value) for key, value in dict(row).items()}


def _serialize_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [_serialize_row(row) for row in rows]


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _embedding_model_singleton():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        cache_dir = os.environ.get("HF_HOME", "/tmp/.cache/huggingface")
        os.environ.setdefault("HF_HOME", cache_dir)
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        os.environ.setdefault("HF_HUB_CACHE", cache_dir)
        logger.info("Loading embedding model %s", EMBEDDING_MODEL_NAME)
        _EMBEDDING_MODEL = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            cache_folder=cache_dir,
        )
    return _EMBEDDING_MODEL


def _embed_query(query: str) -> str:
    model = _embedding_model_singleton()
    vector = model.encode([query], show_progress_bar=False)[0].tolist()
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding model returned {len(vector)} dimensions, expected {EMBEDDING_DIM}"
        )
    return _vector_literal(vector)


def _chunk_text(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]

    chunks: list[str] = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(text), step):
        chunk = text[start : start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
        if start + CHUNK_SIZE >= len(text):
            break
    return chunks


def _embedding_id(document_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{document_id}:{chunk_index}".encode("utf-8")).hexdigest()
    return f"weather-emb:{digest[:32]}"


def _load_unembedded_weather_documents(
    *,
    limit: int,
    source_type: str | None,
) -> list[dict[str, Any]]:
    where_clauses = [
        "length(trim(COALESCE(d.narrative_text, ''))) > 0",
        f"""
        NOT EXISTS (
            SELECT 1
            FROM {WEATHER_EMBEDDINGS_TABLE} e
            WHERE e.document_id = d.id
        )
        """,
    ]
    params: list[Any] = []
    if source_type:
        where_clauses.append("d.source_type = %s")
        params.append(source_type)
    params.append(limit)

    return lakebase.run_query(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.narrative_text,
            d.synced_at
        FROM {WEATHER_DOCUMENTS_TABLE} d
        WHERE {" AND ".join(where_clauses)}
        ORDER BY d.synced_at ASC
        LIMIT %s
        """,
        tuple(params),
    )


def _embed_unembedded_weather_documents(
    *,
    limit: int,
    source_type: str | None,
) -> dict[str, int]:
    documents = _load_unembedded_weather_documents(limit=limit, source_type=source_type)
    chunks: list[dict[str, Any]] = []
    for document in documents:
        for chunk_index, chunk_text in enumerate(_chunk_text(document["narrative_text"])):
            chunks.append(
                {
                    "id": _embedding_id(document["id"], chunk_index),
                    "document_id": document["id"],
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                }
            )

    if not chunks:
        return {
            "documents_selected": len(documents),
            "chunks_prepared": 0,
            "embeddings_inserted": 0,
        }

    model = _embedding_model_singleton()
    vectors = model.encode(
        [chunk["chunk_text"] for chunk in chunks],
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False,
    )

    rows = []
    for chunk, vector in zip(chunks, vectors):
        values = vector.tolist()
        if len(values) != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding model returned {len(values)} dimensions, expected {EMBEDDING_DIM}"
            )
        rows.append(
            (
                chunk["id"],
                chunk["document_id"],
                chunk["chunk_index"],
                chunk["chunk_text"],
                _vector_literal(values),
                EMBEDDING_MODEL_NAME,
            )
        )

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            inserted = execute_values(
                cur,
                f"""
                INSERT INTO {WEATHER_EMBEDDINGS_TABLE} (
                    id,
                    document_id,
                    chunk_index,
                    chunk_text,
                    embedding,
                    model_name,
                    created_at
                )
                VALUES %s
                ON CONFLICT (document_id, chunk_index) DO NOTHING
                RETURNING id
                """,
                rows,
                template="(%s, %s, %s, %s, %s::vector, %s, now())",
                page_size=100,
                fetch=True,
            )
            conn.commit()

    return {
        "documents_selected": len(documents),
        "chunks_prepared": len(chunks),
        "embeddings_inserted": len(inserted),
    }


def _upsert_weather_documents(documents: list[dict[str, Any]]) -> int:
    if not documents:
        return 0

    rows = [
        (
            doc["id"],
            doc["location"],
            doc.get("latitude"),
            doc.get("longitude"),
            doc.get("office"),
            doc.get("grid_x"),
            doc.get("grid_y"),
            doc["source_type"],
            doc.get("headline"),
            doc.get("event"),
            doc["narrative_text"],
            doc.get("issued_at"),
            doc.get("effective_at"),
            doc.get("expires_at"),
            Json(doc["payload"]),
        )
        for doc in documents
    ]

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {WEATHER_DOCUMENTS_TABLE} (
                    id,
                    location,
                    latitude,
                    longitude,
                    office,
                    grid_x,
                    grid_y,
                    source_type,
                    headline,
                    event,
                    narrative_text,
                    issued_at,
                    effective_at,
                    expires_at,
                    payload,
                    synced_at
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE
                    SET location = EXCLUDED.location,
                        latitude = EXCLUDED.latitude,
                        longitude = EXCLUDED.longitude,
                        office = EXCLUDED.office,
                        grid_x = EXCLUDED.grid_x,
                        grid_y = EXCLUDED.grid_y,
                        source_type = EXCLUDED.source_type,
                        headline = EXCLUDED.headline,
                        event = EXCLUDED.event,
                        narrative_text = EXCLUDED.narrative_text,
                        issued_at = EXCLUDED.issued_at,
                        effective_at = EXCLUDED.effective_at,
                        expires_at = EXCLUDED.expires_at,
                        payload = EXCLUDED.payload,
                        synced_at = EXCLUDED.synced_at
                """,
                rows,
                template=(
                    "(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                    "%s, %s, %s, %s, now())"
                ),
                page_size=100,
            )
            conn.commit()
    return len(documents)


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", os.getenv("PORT", "8000")))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host=host, port=port, debug=debug)
