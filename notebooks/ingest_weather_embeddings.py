# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Lakebase pgvector
# MAGIC
# MAGIC This notebook/script reads `weather_documents`, chunks `narrative_text`,
# MAGIC embeds each chunk with `sentence-transformers/all-MiniLM-L6-v2`, and
# MAGIC writes vectors into `weather_embeddings` via psycopg2.

# COMMAND ----------

# MAGIC %pip install -q 'databricks-sdk>=0.30.0' 'psycopg2-binary>=2.9.9' sentence-transformers pandas python-dotenv

# COMMAND ----------

try:
    dbutils.library.restartPython()
except Exception:
    pass

# COMMAND ----------

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


def _project_root() -> Path:
    if "__file__" in globals():
        candidate = Path(__file__).resolve().parents[1]
        if (candidate / "lakebase.py").exists():
            return candidate
    cwd = Path.cwd()
    if (cwd / "lakebase.py").exists():
        return cwd
    if (cwd.parent / "lakebase.py").exists():
        return cwd.parent
    return cwd


ROOT = _project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import lakebase  # noqa: E402


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _safe_identifier(value: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Unsafe SQL identifier: {value!r}")
    return value


def _widget(name: str, default: str, label: str | None = None) -> str:
    try:
        dbutils.widgets.text(name, default, label or name)
        return dbutils.widgets.get(name)
    except Exception:
        return os.environ.get(name.upper(), default)


DOCUMENTS_TABLE = _safe_identifier(
    _widget("documents_table_name", os.environ.get("WEATHER_DOCUMENTS_TABLE", "weather_documents"))
)
EMBEDDINGS_TABLE = _safe_identifier(
    _widget("embeddings_table_name", os.environ.get("WEATHER_EMBEDDINGS_TABLE", "weather_embeddings"))
)
EMBEDDING_MODEL_NAME = _widget(
    "embedding_model",
    os.environ.get("WEATHER_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
)
CHUNK_SIZE = int(_widget("chunk_size", os.environ.get("WEATHER_CHUNK_SIZE", "800")))
CHUNK_OVERLAP = int(_widget("chunk_overlap", os.environ.get("WEATHER_CHUNK_OVERLAP", "100")))
BATCH_LIMIT = int(_widget("batch_limit", os.environ.get("WEATHER_EMBED_BATCH_LIMIT", "500")))
EMBED_BATCH_SIZE = int(_widget("embed_batch_size", os.environ.get("WEATHER_EMBED_BATCH_SIZE", "32")))

MODEL_DIMS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "BAAI/bge-small-en-v1.5": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
}
EMBEDDING_DIM = MODEL_DIMS.get(EMBEDDING_MODEL_NAME)
if EMBEDDING_DIM is None:
    raise ValueError(
        f"Unknown embedding model {EMBEDDING_MODEL_NAME!r}; add its dimension to MODEL_DIMS"
    )
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError("chunk_overlap must be smaller than chunk_size")

print(f"Documents table: {DOCUMENTS_TABLE}")
print(f"Embeddings table: {EMBEDDINGS_TABLE}")
print(f"Embedding model: {EMBEDDING_MODEL_NAME} ({EMBEDDING_DIM} dims)")
print(f"Chunking: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

# COMMAND ----------

lakebase.ensure_weather_schema(
    documents_table=DOCUMENTS_TABLE,
    embeddings_table=EMBEDDINGS_TABLE,
    embedding_dim=EMBEDDING_DIM,
)

# COMMAND ----------


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def _embedding_id(document_id: str, chunk_index: int) -> str:
    digest = hashlib.sha256(f"{document_id}:{chunk_index}".encode("utf-8")).hexdigest()
    return f"weather-emb:{digest[:32]}"


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


documents = lakebase.run_query(
    f"""
    SELECT
        d.id,
        d.location,
        d.source_type,
        d.headline,
        d.narrative_text,
        d.synced_at
    FROM {DOCUMENTS_TABLE} d
    WHERE length(trim(COALESCE(d.narrative_text, ''))) > 0
      AND NOT EXISTS (
          SELECT 1
          FROM {EMBEDDINGS_TABLE} e
          WHERE e.document_id = d.id
            AND e.model_name = %s
      )
    ORDER BY d.synced_at DESC
    LIMIT %s
    """,
    (EMBEDDING_MODEL_NAME, BATCH_LIMIT),
)

print(f"Loaded {len(documents)} unembedded weather document(s)")

chunks: list[dict[str, Any]] = []
for document in documents:
    for chunk_index, chunk_text in enumerate(
        _chunk_text(document["narrative_text"], CHUNK_SIZE, CHUNK_OVERLAP)
    ):
        chunks.append(
            {
                "id": _embedding_id(document["id"], chunk_index),
                "document_id": document["id"],
                "chunk_index": chunk_index,
                "chunk_text": chunk_text,
            }
        )

print(f"Prepared {len(chunks)} chunk(s)")

# COMMAND ----------

if chunks:
    import os as _os

    from sentence_transformers import SentenceTransformer

    cache_dir = _os.environ.get("HF_HOME", "/tmp/.cache/huggingface")
    _os.environ.setdefault("HF_HOME", cache_dir)
    _os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    _os.environ.setdefault("HF_HUB_CACHE", cache_dir)

    print(f"Loading {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder=cache_dir)

    texts = [chunk["chunk_text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunk(s)")
    vectors = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
    )

    rows = []
    for chunk, vector in zip(chunks, vectors):
        values = vector.tolist()
        if len(values) != EMBEDDING_DIM:
            raise ValueError(
                f"Model returned {len(values)} dims, expected {EMBEDDING_DIM}"
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

    from psycopg2.extras import execute_values

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO {EMBEDDINGS_TABLE} (
                    id,
                    document_id,
                    chunk_index,
                    chunk_text,
                    embedding,
                    model_name,
                    created_at
                )
                VALUES %s
                ON CONFLICT (document_id, chunk_index) DO UPDATE
                    SET chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        model_name = EXCLUDED.model_name,
                        created_at = EXCLUDED.created_at
                """,
                rows,
                template="(%s, %s, %s, %s, %s::vector, %s, now())",
                page_size=100,
            )
            conn.commit()
            print(f"Upserted {len(rows)} weather embedding row(s)")
else:
    print("No new weather documents need embeddings")

# COMMAND ----------

counts = lakebase.run_query(
    f"""
    SELECT
        (SELECT COUNT(*)::int FROM {DOCUMENTS_TABLE}) AS documents,
        (SELECT COUNT(*)::int FROM {EMBEDDINGS_TABLE}) AS embeddings
    """
)[0]
print(f"Weather documents: {counts['documents']}")
print(f"Weather embeddings: {counts['embeddings']}")
