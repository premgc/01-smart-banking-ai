from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from functools import lru_cache

from dotenv import load_dotenv
from openai import AzureOpenAI
from databricks.vector_search.client import VectorSearchClient

# =========================================================
# LOGGING
# =========================================================
logger = logging.getLogger(__name__)

# =========================================================
# LOAD ENV
# =========================================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# =========================================================
# ENV CONFIG
# =========================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "").strip()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()

VECTOR_SEARCH_ENDPOINT = os.getenv(
    "DATABRICKS_VECTOR_SEARCH_ENDPOINT",
    "banking-vs-endpoint"
).strip()

VECTOR_SEARCH_INDEX = os.getenv(
    "DATABRICKS_VECTOR_SEARCH_INDEX",
    "bronze.banking.banking_txn_idx"
).strip()

DEFAULT_LIMIT = int(os.getenv("VECTOR_SEARCH_TOP_K", "5"))

RETURN_COLUMNS = [
    "transaction_key",
    "content",
    "transaction_date",
    "value_date",
    "tran_type",
    "category",
    "type",
    "amount",
    "withdrawal",
    "deposit",
    "balance",
    "description",
    "source_file",
]

# =========================================================
# VALIDATION
# =========================================================
def validate_config() -> None:
    missing = []

    required = {
        "AZURE_OPENAI_ENDPOINT": AZURE_OPENAI_ENDPOINT,
        "AZURE_OPENAI_API_KEY": AZURE_OPENAI_API_KEY,
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        "DATABRICKS_HOST": DATABRICKS_HOST,
        "DATABRICKS_TOKEN": DATABRICKS_TOKEN,
        "DATABRICKS_VECTOR_SEARCH_ENDPOINT": VECTOR_SEARCH_ENDPOINT,
        "DATABRICKS_VECTOR_SEARCH_INDEX": VECTOR_SEARCH_INDEX,
    }

    for key, value in required.items():
        if not value:
            missing.append(key)

    if missing:
        raise RuntimeError(f"Missing configuration: {', '.join(missing)}")


# =========================================================
# OPENAI CLIENT
# =========================================================
@lru_cache(maxsize=1)
def get_openai_client() -> AzureOpenAI:
    validate_config()

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


# =========================================================
# VECTOR SEARCH CLIENT
# =========================================================
@lru_cache(maxsize=1)
def get_vector_search_client() -> VectorSearchClient:
    """
    Supports both old and new Databricks Vector Search SDK versions.
    """
    validate_config()

    try:
        logger.info("Initializing VectorSearchClient using explicit constructor auth")

        return VectorSearchClient(
            workspace_url=DATABRICKS_HOST,
            personal_access_token=DATABRICKS_TOKEN,
            disable_notice=True,
        )

    except TypeError:
        logger.warning(
            "Older Databricks Vector Search SDK detected. "
            "Falling back to environment-based auth."
        )

        os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
        os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

        return VectorSearchClient()


@lru_cache(maxsize=1)
def get_vector_index():
    client = get_vector_search_client()

    return client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_SEARCH_INDEX,
    )


# =========================================================
# EMBEDDINGS
# =========================================================
def get_embedding(text: str) -> List[float]:
    if not text or not text.strip():
        raise ValueError("Text is required for embedding")

    client = get_openai_client()

    response = client.embeddings.create(
        input=text.strip(),
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )

    return response.data[0].embedding


# =========================================================
# HEALTH CHECK
# =========================================================
def health_check() -> tuple[bool, str]:
    try:
        index = get_vector_index()

        index.similarity_search(
            query_vector=get_embedding("health check"),
            columns=["content"],
            num_results=1,
        )

        return True, "Databricks Vector Search ready"

    except Exception as e:
        logger.exception("Databricks Vector Search health check failed")
        return False, f"Databricks Vector Search error: {str(e)}"


# =========================================================
# RAW SEARCH
# =========================================================
def search_raw(query: str, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    if not query or not query.strip():
        return {"result": {"data_array": []}}

    index = get_vector_index()
    query_vector = get_embedding(query)

    return index.similarity_search(
        query_vector=query_vector,
        columns=RETURN_COLUMNS,
        num_results=limit,
    )


# =========================================================
# MAIN SEARCH
# =========================================================
def search(query: str, limit: int = DEFAULT_LIMIT) -> List[str]:
    try:
        raw_results = search_raw(query=query, limit=limit)

        data_array = raw_results.get("result", {}).get("data_array", [])

        if not data_array:
            return []

        columns = raw_results.get("manifest", {}).get("columns", [])
        column_names = [col.get("name") for col in columns]

        if "content" not in column_names:
            logger.warning("Content column missing in vector search results")
            return []

        content_index = column_names.index("content")

        return [
            str(row[content_index])
            for row in data_array
            if len(row) > content_index and row[content_index]
        ]

    except Exception as e:
        logger.exception("Vector search failed")
        raise RuntimeError(f"Databricks Vector Search failed: {str(e)}")


# =========================================================
# UPSERT DISABLED
# =========================================================
def upsert_texts(texts: List[str]) -> int:
    raise NotImplementedError(
        "Direct upsert disabled. Load data into Delta table and sync index."
    )