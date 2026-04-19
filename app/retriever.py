from __future__ import annotations

import os
import uuid
from typing import List
from pathlib import Path
from azure.search.documents.models import VectorizedQuery


from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

from openai import AzureOpenAI
from dotenv import load_dotenv


# =========================================================
# LOAD ENV (robust)
# =========================================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# =========================================================
# ENV CONFIG
# =========================================================
SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX", "transactions-index")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")


# =========================================================
# VALIDATION (fail fast)
# =========================================================
if not SEARCH_ENDPOINT or not SEARCH_KEY:
    raise RuntimeError("Azure Search configuration missing")

if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_DEPLOYMENT:
    raise RuntimeError("Azure OpenAI configuration missing")


# =========================================================
# CLIENTS
# =========================================================
def get_search_client() -> SearchClient:
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
    )


openai_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,   # ✅ CORRECT NAME
    api_version="2024-02-01"
)

# =========================================================
# EMBEDDING (AZURE OPENAI SDK)
# =========================================================

def get_embedding(text: str):
    response = openai_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )
    return response.data[0].embedding
# =========================================================
# HEALTH CHECK
# =========================================================
def health_check() -> tuple[bool, str]:
    try:
        client = get_search_client()
        _ = list(client.search(search_text="test", top=1))
        return True, "Azure AI Search ready"
    except Exception as e:
        return False, f"Azure Search error: {e}"


# =========================================================
# UPSERT DATA
# =========================================================
def upsert_texts(texts: List[str]) -> int:
    client = get_search_client()

    clean_texts = [t.strip() for t in texts if t and t.strip()]
    if not clean_texts:
        return 0

    docs = []
    for text in clean_texts:
        docs.append({
            "id": str(uuid.uuid4()),
            "content": text,
            "embedding": get_embedding(text),
        })

    result = client.upload_documents(documents=docs)

    success_count = sum(1 for r in result if r.succeeded)

    print(f"✅ Uploaded {success_count} documents to Azure AI Search")

    return success_count


# =========================================================
# SEARCH (VECTOR + TEXT HYBRID)
# =========================================================
def search(query: str, limit: int = 5):
    client = get_search_client()

    vector = get_embedding(query)

    vector_query = VectorizedQuery(
        vector=vector,
        k_nearest_neighbors=limit,
        fields="embedding"
    )

    results = client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=limit,
    )

    return [doc["content"] for doc in results]