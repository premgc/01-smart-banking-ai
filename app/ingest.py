import os
import hashlib
from io import StringIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

# =====================================================
# LOAD ENV
# =====================================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# =====================================================
# AZURE CONFIG
# =====================================================
BLOB_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONNECTION_STRING")
BLOB_CONTAINER = os.getenv("AZURE_BLOB_CONTAINER")
BLOB_FILE = os.getenv("AZURE_BLOB_FILE", "statement.csv")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# =====================================================
# INGESTION BEHAVIOR
# =====================================================
HARD_DELETE_MISSING = os.getenv("HARD_DELETE_MISSING", "false").lower() == "true"
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
UPLOAD_BATCH_SIZE = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))

# =====================================================
# VALIDATION
# =====================================================
def validate_config():
    missing = []

    if not BLOB_CONNECTION_STRING:
        missing.append("AZURE_BLOB_CONNECTION_STRING")
    if not BLOB_CONTAINER:
        missing.append("AZURE_BLOB_CONTAINER")
    if not SEARCH_ENDPOINT:
        missing.append("AZURE_SEARCH_ENDPOINT")
    if not SEARCH_KEY:
        missing.append("AZURE_SEARCH_KEY")
    if not SEARCH_INDEX:
        missing.append("AZURE_SEARCH_INDEX")
    if not OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not OPENAI_KEY:
        missing.append("AZURE_OPENAI_API_KEY")
    if not EMBEDDING_MODEL:
        missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    if missing:
        raise RuntimeError(f"Missing required config: {', '.join(missing)}")

# =====================================================
# CLIENTS
# =====================================================
def get_blob_service():
    return BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)

def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

def get_index_client():
    return SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

def get_openai_client():
    return AzureOpenAI(
        api_key=OPENAI_KEY,
        api_version="2024-02-15-preview",
        azure_endpoint=OPENAI_ENDPOINT,
    )

# =====================================================
# CREATE INDEX IF MISSING
# =====================================================
def create_index_if_not_exists():
    print(f"🔎 Checking if index exists: {SEARCH_INDEX}")

    index_client = get_index_client()
    existing_indexes = [idx.name for idx in index_client.list_indexes()]

    if SEARCH_INDEX in existing_indexes:
        print(f"✅ Index already exists: {SEARCH_INDEX}")
        return

    print(f"🚀 Creating index: {SEARCH_INDEX}")

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="tran_type",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="date",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="category",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="type",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SimpleField(
            name="amount",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="description",
            type=SearchFieldDataType.String,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            retrievable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(name="hnsw-config")
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config",
            )
        ],
    )

    index = SearchIndex(
        name=SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
    )

    index_client.create_index(index)
    print(f"✅ Index created: {SEARCH_INDEX}")

# =====================================================
# STEP 1 — READ FROM AZURE BLOB
# =====================================================
def load_csv_from_blob() -> pd.DataFrame:
    print(f"📥 Downloading {BLOB_FILE} from Azure Blob...")

    blob_service = get_blob_service()
    blob_client = blob_service.get_blob_client(
        container=BLOB_CONTAINER,
        blob=BLOB_FILE,
    )

    blob_data = blob_client.download_blob().readall()
    text = blob_data.decode("utf-8", errors="ignore")

    df = pd.read_csv(StringIO(text), skiprows=10)
    df = df.dropna(how="all")

    print(f"✅ Loaded {len(df)} raw rows from Blob")
    print("🔥 Raw columns:", df.columns.tolist())

    return df

# =====================================================
# HELPERS
# =====================================================
def clean_money_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.strip()
        .replace("", "0")
        .replace("nan", "0"),
        errors="coerce",
    ).fillna(0)

def derive_category(description: str) -> str:
    desc = str(description).lower()

    if "upi" in desc:
        return "upi"
    if "nft/" in desc or "neft" in desc:
        return "neft"
    if "imps" in desc:
        return "imps"
    if "cash" in desc or "atm" in desc:
        return "cash"
    if "salary" in desc:
        return "salary"
    if "charge" in desc or "chrg" in desc or "fee" in desc:
        return "bank charges"
    if "amazon" in desc or "flipkart" in desc:
        return "shopping"
    if "swiggy" in desc or "zomato" in desc or "restaurant" in desc:
        return "food"
    if "uber" in desc or "ola" in desc or "fuel" in desc:
        return "transport"
    return "other"

def derive_tran_type(description: str, existing_value: str) -> str:
    existing = str(existing_value).strip().upper() if existing_value is not None else ""
    if existing and existing != "NAN":
        return existing

    desc = str(description).upper()

    if "UPI" in desc:
        return "UPI"
    if "NEFT" in desc or "NFT/" in desc:
        return "NEFT"
    if "IMPS" in desc:
        return "IMPS"
    if "ATM" in desc or "CASH" in desc:
        return "CASH"
    return "OTHER"

def normalize_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

# =====================================================
# STEP 2 — CLEAN + NORMALIZE
# =====================================================
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]

    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    print("✅ Normalized columns:", df.columns.tolist())

    rename_map = {
        "sl. no.": "sl_no",
        "tran date": "date",
        "particulars": "description",
        "value date": "value_date",
        "tran type": "tran_type",
        "cheque details": "cheque_details",
        "withdrawal": "withdrawal",
        "deposit": "deposit",
        "balance amount": "balance",
    }

    df = df.rename(columns=rename_map)

    required_cols = ["date", "description", "withdrawal", "deposit"]
    for col in required_cols:
        if col not in df.columns:
            raise Exception(f"❌ Missing column: {col}")

    df = df[df["date"].notna()]
    df = df[df["description"].notna()]

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

    if "value_date" in df.columns:
        df["value_date"] = pd.to_datetime(df["value_date"], errors="coerce", dayfirst=True)
    else:
        df["value_date"] = pd.NaT

    df["withdrawal"] = clean_money_series(df["withdrawal"])
    df["deposit"] = clean_money_series(df["deposit"])

    if "balance" in df.columns:
        df["balance"] = clean_money_series(df["balance"])
    else:
        df["balance"] = 0.0

    df["amount"] = df["deposit"] - df["withdrawal"]
    df["type"] = df["amount"].apply(lambda x: "credit" if x > 0 else "debit")
    df["category"] = df["description"].apply(derive_category)

    df["description"] = df["description"].astype(str).str.strip()

    if "tran_type" not in df.columns:
        df["tran_type"] = ""

    df["tran_type"] = df.apply(
        lambda r: derive_tran_type(r["description"], r.get("tran_type")),
        axis=1
    )

    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df["transaction_key"] = df.apply(build_transaction_key, axis=1)
    before = len(df)
    df = df.drop_duplicates(subset=["transaction_key"]).reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"♻️ Removed {before - after} duplicate source rows")

    print(f"✅ Cleaned rows: {len(df)}")
    return df

# =====================================================
# STEP 3 — STABLE BUSINESS KEY / CDC KEY
# =====================================================
def build_transaction_key(row: pd.Series) -> str:
    parts = [
        row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "",
        row["value_date"].strftime("%Y-%m-%d") if pd.notna(row["value_date"]) else "",
        normalize_text(row.get("description", "")),
        normalize_text(row.get("tran_type", "")),
        f"{float(row.get('withdrawal', 0.0)):.2f}",
        f"{float(row.get('deposit', 0.0)):.2f}",
        f"{float(row.get('balance', 0.0)):.2f}",
    ]
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def build_document_id(row: pd.Series) -> str:
    return build_transaction_key(row)

# =====================================================
# STEP 4 — FORMAT TEXT FOR SEARCH / EMBEDDING
# =====================================================
def format_row(row: pd.Series) -> str:
    value_date = row["value_date"].strftime("%Y-%m-%d") if pd.notna(row.get("value_date")) else ""
    balance_text = f"\nBalance: {row['balance']}" if pd.notna(row.get("balance")) else ""
    tran_type_text = f"\nTransaction Type: {row['tran_type']}" if row.get("tran_type") else ""

    return (
        f"Transaction Date: {row['date'].strftime('%Y-%m-%d')}\n"
        f"Value Date: {value_date}\n"
        f"Type: {row['type']}\n"
        f"Amount: {row['amount']}\n"
        f"Category: {row['category']}\n"
        f"Description: {row['description']}"
        f"{tran_type_text}"
        f"{balance_text}"
    )

# =====================================================
# STEP 5 — LOAD EXISTING INDEX DOCS (FOR CDC)
# =====================================================
def load_existing_documents() -> Dict[str, str]:
    print("🔎 Loading existing documents from Azure Search for CDC check...")

    search_client = get_search_client()
    existing_docs: Dict[str, str] = {}

    try:
        results = search_client.search(
            search_text="*",
            select=["id", "content"],
            top=1000,
        )

        count = 0
        for doc in results:
            doc_id = doc.get("id")
            content = doc.get("content", "")
            if doc_id:
                existing_docs[doc_id] = content
                count += 1

        print(f"✅ Existing indexed docs found: {count}")
        return existing_docs

    except Exception as e:
        if "not found" in str(e).lower():
            print("ℹ️ Index not found or empty yet. Treating as first load.")
            return {}
        raise

# =====================================================
# STEP 6 — PLAN CHANGES
# =====================================================
def plan_changes(df: pd.DataFrame, existing_docs: Dict[str, str]) -> Tuple[List[dict], List[str], int]:
    to_upsert = []
    current_ids = []
    unchanged_count = 0

    for _, row in df.iterrows():
        doc_id = build_document_id(row)
        content = format_row(row)

        current_ids.append(doc_id)

        if doc_id in existing_docs and existing_docs[doc_id] == content:
            unchanged_count += 1
            continue

        to_upsert.append({
            "id": doc_id,
            "content": content,
            "tran_type": row["tran_type"],
            "date": row["date"].strftime("%Y-%m-%d"),
            "category": row["category"],
            "amount": float(row["amount"]),
            "type": row["type"],
            "description": row["description"],
        })

    return to_upsert, current_ids, unchanged_count

# =====================================================
# STEP 7 — BATCH EMBEDDINGS
# =====================================================
def add_embeddings_in_batches(docs: List[dict]) -> List[dict]:
    if not docs:
        return docs

    print(f"🧠 Generating embeddings for {len(docs)} changed/new documents...")

    openai_client = get_openai_client()

    for i in range(0, len(docs), EMBED_BATCH_SIZE):
        batch = docs[i:i + EMBED_BATCH_SIZE]
        texts = [d["content"] for d in batch]

        response = openai_client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )

        for doc, emb in zip(batch, response.data):
            doc["embedding"] = emb.embedding

        print(
            f"✅ Embedded batch {i // EMBED_BATCH_SIZE + 1} "
            f"- {len(batch)}/{len(batch)} succeeded"
        )

    return docs

# =====================================================
# STEP 8 — UPSERT TO AZURE SEARCH
# =====================================================
def upsert_documents(docs: List[dict]):
    if not docs:
        print("ℹ️ No new or changed documents to upload")
        return

    print("📤 Upserting documents to Azure Search...")

    search_client = get_search_client()

    for i in range(0, len(docs), UPLOAD_BATCH_SIZE):
        batch = docs[i:i + UPLOAD_BATCH_SIZE]
        result = search_client.merge_or_upload_documents(documents=batch)

        success_count = sum(1 for r in result if r.succeeded)
        print(
            f"✅ Upserted batch {i // UPLOAD_BATCH_SIZE + 1} "
            f"- {success_count}/{len(batch)} succeeded"
        )

    print("✅ Upsert completed")

# =====================================================
# STEP 9 — OPTIONAL HARD DELETE
# =====================================================
def delete_missing_documents(current_ids: List[str]):
    if not HARD_DELETE_MISSING:
        print("ℹ️ HARD_DELETE_MISSING=false, skipping delete phase")
        return

    print("🗑️ HARD_DELETE_MISSING=true, checking for removed transactions...")

    search_client = get_search_client()

    existing_ids = set()
    results = search_client.search(
        search_text="*",
        select=["id"],
        top=1000
    )
    for doc in results:
        if doc.get("id"):
            existing_ids.add(doc["id"])

    current_ids_set = set(current_ids)
    ids_to_delete = existing_ids - current_ids_set

    if not ids_to_delete:
        print("✅ No stale documents to delete")
        return

    delete_docs = [{"id": doc_id} for doc_id in ids_to_delete]

    for i in range(0, len(delete_docs), UPLOAD_BATCH_SIZE):
        batch = delete_docs[i:i + UPLOAD_BATCH_SIZE]
        result = search_client.delete_documents(documents=batch)
        success_count = sum(1 for r in result if r.succeeded)
        print(
            f"✅ Deleted batch {i // UPLOAD_BATCH_SIZE + 1} "
            f"- {success_count}/{len(batch)} succeeded"
        )

    print(f"✅ Deleted {len(ids_to_delete)} stale documents")

# =====================================================
# MAIN
# =====================================================
def main():
    start = datetime.now()

    validate_config()
    create_index_if_not_exists()

    df = load_csv_from_blob()
    df = clean_data(df)

    print("🔥 Sample cleaned data:")
    print(df[["date", "description", "tran_type", "withdrawal", "deposit", "amount", "type", "category"]].head())

    existing_docs = load_existing_documents()
    docs_to_upsert, current_ids, unchanged_count = plan_changes(df, existing_docs)

    print(f"📊 Unchanged docs skipped: {unchanged_count}")
    print(f"📊 New/changed docs to process: {len(docs_to_upsert)}")

    docs_to_upsert = add_embeddings_in_batches(docs_to_upsert)
    upsert_documents(docs_to_upsert)
    delete_missing_documents(current_ids)

    print(f"🚀 Incremental ingestion complete in {datetime.now() - start}")

if __name__ == "__main__":
    main()