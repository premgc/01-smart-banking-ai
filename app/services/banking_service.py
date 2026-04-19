import os
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


def _get_search_client() -> SearchClient:
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT", "").strip()
    index_name = os.getenv("AZURE_SEARCH_INDEX", "").strip()
    api_key = os.getenv("AZURE_SEARCH_KEY", "").strip()

    missing = []
    if not endpoint:
        missing.append("AZURE_SEARCH_ENDPOINT")
    if not index_name:
        missing.append("AZURE_SEARCH_INDEX")
    if not api_key:
        missing.append("AZURE_SEARCH_KEY")

    if missing:
        raise RuntimeError(f"Missing Azure Search config: {', '.join(missing)}")

    return SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )


# ======================================================
# QUERY HELPERS
# ======================================================
def _extract_tran_type_from_query(query: str) -> Optional[str]:
    q = query.lower()

    if "upi" in q:
        return "UPI"
    if "neft" in q or "nft" in q:
        return "NEFT"
    if "imps" in q:
        return "IMPS"
    if "atm" in q:
        return "ATM"
    if "cash" in q:
        return "CASH"
    if "tfr" in q or "transfer" in q:
        return "TFR"

    return None


def _extract_date_filter(query: str) -> Optional[str]:
    q = query.lower()
    today = datetime.today().date()

    if "last 7 days" in q or "last seven days" in q:
        start = today - timedelta(days=7)
        return f"date ge '{start.isoformat()}'"

    if "last 30 days" in q or "last thirty days" in q:
        start = today - timedelta(days=30)
        return f"date ge '{start.isoformat()}'"

    if "this month" in q:
        start = today.replace(day=1)
        return f"date ge '{start.isoformat()}'"

    return None


def _combine_filters(filters: List[str]) -> Optional[str]:
    clean = [f for f in filters if f]
    if not clean:
        return None
    return " and ".join(clean)


# ======================================================
# TRANSACTION TYPES
# ======================================================
def get_all_tran_types() -> List[str]:
    try:
        client = _get_search_client()

        results = client.search(
            search_text="*",
            facets=["tran_type,count:20"],
            top=0,
        )

        facets = results.get_facets()
        if not facets or "tran_type" not in facets:
            return []

        values = []
        for item in facets["tran_type"]:
            value = item.get("value")
            if value:
                values.append(str(value).strip())

        return sorted(set(values))

    except Exception:
        logger.exception("Error fetching transaction types")
        return []


def get_all_tran_types_with_count() -> List[Tuple[str, int]]:
    try:
        client = _get_search_client()

        results = client.search(
            search_text="*",
            facets=["tran_type,count:20"],
            top=0,
        )

        facets = results.get_facets()
        if not facets or "tran_type" not in facets:
            return []

        values: List[Tuple[str, int]] = []
        for item in facets["tran_type"]:
            value = item.get("value")
            count = item.get("count", 0)
            if value:
                values.append((str(value).strip(), int(count)))

        return sorted(values, key=lambda x: (-x[1], x[0]))

    except Exception:
        logger.exception("Error fetching transaction types with count")
        return []


# ======================================================
# SEARCH TRANSACTIONS
# ======================================================
def search_transactions(query: str) -> List[Dict]:
    try:
        client = _get_search_client()

        tran_type = _extract_tran_type_from_query(query)
        date_filter = _extract_date_filter(query)

        filters = []
        if tran_type:
            filters.append(f"tran_type eq '{tran_type}'")
        if date_filter:
            filters.append(date_filter)

        filter_query = _combine_filters(filters)

        results = client.search(
            search_text="*",
            filter=filter_query,
            top=20,
            select=[
                "date",
                "description",
                "tran_type",
                "amount",
                "type",
                "category",
            ],
            order_by=["date desc"]
        )

        return [dict(r) for r in results]

    except Exception:
        logger.exception("Search error")
        return []


# ======================================================
# HYBRID / VECTOR-READY SEARCH
# ======================================================
def search_transactions_hybrid(query: str) -> List[Dict]:
    """
    Placeholder hybrid search path.
    Right now it falls back to structured search.
    Later you can plug embeddings/vector query here.
    """
    try:
        return search_transactions(query)
    except Exception:
        logger.exception("Hybrid search error")
        return []


# ======================================================
# CATEGORY SPEND
# ======================================================
def get_spending_by_category(query: str) -> List[Tuple[str, float]]:
    try:
        client = _get_search_client()

        date_filter = _extract_date_filter(query)
        type_filter = "type eq 'debit'"

        filter_query = _combine_filters([type_filter, date_filter])

        results = client.search(
            search_text="*",
            filter=filter_query,
            top=1000,
            select=["category", "amount", "type"]
        )

        totals: Dict[str, float] = {}

        for r in results:
            category = (r.get("category") or "other").strip()
            amount = float(r.get("amount") or 0)

            # debit amounts may be negative from ingestion
            spend = abs(amount)
            totals[category] = totals.get(category, 0.0) + spend

        return sorted(totals.items(), key=lambda x: x[1], reverse=True)

    except Exception:
        logger.exception("Error getting spending by category")
        return []


# ======================================================
# TOP LOSSES / INSIGHTS
# ======================================================
def get_top_expenses(query: str) -> List[Dict]:
    try:
        client = _get_search_client()

        date_filter = _extract_date_filter(query)
        type_filter = "type eq 'debit'"
        filter_query = _combine_filters([type_filter, date_filter])

        results = client.search(
            search_text="*",
            filter=filter_query,
            top=20,
            select=["date", "description", "tran_type", "amount", "category", "type"],
            order_by=["amount asc"]  # most negative first
        )

        return [dict(r) for r in results]

    except Exception:
        logger.exception("Error getting top expenses")
        return []


def get_financial_insights(query: str) -> Dict:
    try:
        category_spend = get_spending_by_category(query)
        top_expenses = get_top_expenses(query)

        total_spend = sum(amount for _, amount in category_spend)
        top_category = category_spend[0] if category_spend else None

        return {
            "total_spend": total_spend,
            "top_category": top_category,
            "top_expenses": top_expenses[:5],
            "category_spend": category_spend[:5],
        }

    except Exception:
        logger.exception("Error building financial insights")
        return {
            "total_spend": 0.0,
            "top_category": None,
            "top_expenses": [],
            "category_spend": [],
        }