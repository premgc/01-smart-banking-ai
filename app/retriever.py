from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

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

# Azure OpenAI embedding config
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01").strip()

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    ""
).strip()

# Databricks Vector Search config
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

# Columns to return from Databricks Vector Search
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

    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")

    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY")

    if not AZURE_OPENAI_EMBEDDING_DEPLOYMENT:
        missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if not DATABRICKS_HOST:
        missing.append("DATABRICKS_HOST")

    if not DATABRICKS_TOKEN:
        missing.append("DATABRICKS_TOKEN")

    if not VECTOR_SEARCH_ENDPOINT:
        missing.append("DATABRICKS_VECTOR_SEARCH_ENDPOINT")

    if not VECTOR_SEARCH_INDEX:
        missing.append("DATABRICKS_VECTOR_SEARCH_INDEX")

    if missing:
        raise RuntimeError(f"Missing configuration: {', '.join(missing)}")


# =========================================================
# CLIENTS
# =========================================================
def get_openai_client() -> AzureOpenAI:
    validate_config()

    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def get_vector_search_client() -> VectorSearchClient:
    validate_config()

    return VectorSearchClient(
        workspace_url=DATABRICKS_HOST,
        personal_access_token=DATABRICKS_TOKEN,
        disable_notice=True,
    )


def get_vector_index():
    client = get_vector_search_client()

    return client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT,
        index_name=VECTOR_SEARCH_INDEX,
    )


# =========================================================
# EMBEDDING
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

        result = index.similarity_search(
            query_vector=get_embedding("health check"),
            columns=["content"],
            num_results=1,
        )

        return True, "Databricks Vector Search ready"

    except Exception as e:
        logger.exception("Databricks Vector Search health check failed")
        return False, f"Databricks Vector Search error: {str(e)}"


# =========================================================
# SEARCH DATBRICKS VECTOR INDEX
# =========================================================
def search_raw(query: str, limit: int = DEFAULT_LIMIT) -> Dict[str, Any]:
    """
    Returns raw Databricks Vector Search response.
    Useful for debugging.
    """
    if not query or not query.strip():
        return {"result": {"data_array": []}}

    index = get_vector_index()
    query_vector = get_embedding(query)

    results = index.similarity_search(
        query_vector=query_vector,
        columns=RETURN_COLUMNS,
        num_results=limit,
    )

    return results


def search(query: str, limit: int = DEFAULT_LIMIT) -> List[str]:
    """
    Main function used by the rest of your app.

    Keeps the old contract:
        search(query) -> List[str]

    So banking_service.py / routes should not break.
    """
    try:
        raw_results = search_raw(query=query, limit=limit)

        data_array = (
            raw_results
            .get("result", {})
            .get("data_array", [])
        )

        if not data_array:
            return []

        columns = raw_results.get("manifest", {}).get("columns", [])
        column_names = [col.get("name") for col in columns]

        content_index = column_names.index("content") if "content" in column_names else None

        contents = []

        for row in data_array:
            if content_index is not None and len(row) > content_index:
                content = row[content_index]
                if content:
                    contents.append(str(content))

        return contents

    except Exception as e:
        logger.exception("Vector search failed")
        raise RuntimeError(f"Databricks Vector Search failed: {str(e)}")


# =========================================================
# UPSERT PLACEHOLDER
# =========================================================
def upsert_texts(texts: List[str]) -> int:
    """
    Databricks Vector Search is now backed by your Delta table.

    Do not upload documents directly here like Azure AI Search.

    Data should be inserted/merged into:
        bronze.banking.statementpdf_transactions_vector

    Then sync the Vector Search index.
    """
    raise NotImplementedError(
        "Direct upsert is disabled. "
        "Load data into the Delta table, then sync the Databricks Vector Search index."
    )