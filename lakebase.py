"""
Lakebase (Databricks-managed Postgres) connection helper.

The deployed Databricks App reads one secret, database/lakebase-url, that holds
the full Postgres connection URL for a native-password Lakebase role.
"""

from __future__ import annotations

import base64
import os
import re
from contextlib import contextmanager

import psycopg2
from databricks.sdk import WorkspaceClient
from psycopg2.extras import RealDictCursor
from sqlalchemy import create_engine

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")
_w: WorkspaceClient | None = None
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _workspace_client() -> WorkspaceClient:
    global _w
    if _w is None:
        _w = WorkspaceClient()
    return _w


def _decode_secret(value: str) -> str:
    try:
        return base64.b64decode(value).decode("utf-8")
    except Exception:
        return value


def _lakebase_url() -> str:
    direct_url = os.environ.get("LAKEBASE_URL")
    if direct_url:
        return direct_url

    secret = _workspace_client().secrets.get_secret(scope=_SCOPE, key=_KEY)
    return _decode_secret(secret.value)


@contextmanager
def get_connection():
    """Yield a raw psycopg2 connection with dict-like rows."""
    conn = psycopg2.connect(_lakebase_url(), cursor_factory=RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def get_engine():
    """Return a SQLAlchemy engine for Lakebase."""
    return create_engine(_lakebase_url())


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query against Lakebase and return rows as list[dict]."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE/DDL statement and return affected rows."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


def _safe_identifier(name: str) -> str:
    """Validate an env-provided table/index name before interpolating SQL."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return name


def _run_optional_schema_statement(sql: str) -> None:
    """Run optional DDL that may be unavailable to a non-owner app role."""
    try:
        run_write(sql)
    except Exception as exc:
        message = str(exc).lower()
        if "must be owner" in message or "permission denied" in message:
            return
        raise


def ensure_weather_schema(
    documents_table: str | None = None,
    embeddings_table: str | None = None,
    embedding_dim: int = 384,
) -> None:
    """Create the Weather Intelligence Lakebase tables and pgvector index.

    Raw weather documents live in one table, while chunk-level vectors live in
    a separate pgvector table.
    """
    documents_table = _safe_identifier(
        documents_table or os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents")
    )
    embeddings_table = _safe_identifier(
        embeddings_table or os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings")
    )
    embedding_dim = int(embedding_dim)
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be a positive integer")

    _run_optional_schema_statement("CREATE EXTENSION IF NOT EXISTS vector")

    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {documents_table} (
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
        )
        """
    )

    run_write(
        f"""
        CREATE TABLE IF NOT EXISTS {embeddings_table} (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL
                REFERENCES {documents_table}(id) ON DELETE CASCADE,
            chunk_index INT NOT NULL,
            chunk_text TEXT NOT NULL CHECK (length(trim(chunk_text)) > 0),
            embedding VECTOR({embedding_dim}) NOT NULL,
            model_name TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (document_id, chunk_index)
        )
        """
    )

    _run_optional_schema_statement(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{documents_table}_location
            ON {documents_table} (location)
        """
    )
    _run_optional_schema_statement(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{documents_table}_source_effective
            ON {documents_table} (source_type, effective_at DESC)
        """
    )
    _run_optional_schema_statement(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{embeddings_table}_document_id
            ON {embeddings_table} (document_id)
        """
    )
    _run_optional_schema_statement(
        f"""
        CREATE INDEX IF NOT EXISTS idx_{embeddings_table}_embedding_hnsw
            ON {embeddings_table}
            USING hnsw (embedding vector_cosine_ops)
        """
    )
