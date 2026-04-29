from __future__ import annotations

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

from databricks import sql

from app.retriever import search_raw

logger = logging.getLogger(__name__)


# ======================================================
# ENV CONFIG
# ======================================================
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").strip()
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "").strip()
DATABRICKS_SQL_HTTP_PATH = os.getenv("DATABRICKS_SQL_HTTP_PATH", "").strip()

DATABRICKS_TABLE = os.getenv(
    "DATABRICKS_TRANSACTION_TABLE",
    "bronze.banking.statementpdf_transactions_vector"
).strip()


# ======================================================
# CONFIG HELPERS
# ======================================================
def _server_hostname() -> str:
    return (
        DATABRICKS_HOST
        .replace("https://", "")
        .replace("http://", "")
        .rstrip("/")
    )


def _validate_sql_config() -> None:
    missing = []

    if not DATABRICKS_HOST:
        missing.append("DATABRICKS_HOST")
    if not DATABRICKS_TOKEN:
        missing.append("DATABRICKS_TOKEN")
    if not DATABRICKS_SQL_HTTP_PATH:
        missing.append("DATABRICKS_SQL_HTTP_PATH")

    if missing:
        raise RuntimeError(f"Missing Databricks SQL config: {', '.join(missing)}")


def _get_sql_connection():
    _validate_sql_config()

    return sql.connect(
        server_hostname=_server_hostname(),
        http_path=DATABRICKS_SQL_HTTP_PATH,
        access_token=DATABRICKS_TOKEN,
    )


def _run_sql(query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict]:
    try:
        with _get_sql_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params or {})
                columns = [c[0] for c in cursor.description]
                rows = cursor.fetchall()

        return [dict(zip(columns, row)) for row in rows]

    except Exception:
        logger.exception("Databricks SQL execution failed")
        return []


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


def _extract_start_date(query: str) -> Optional[str]:
    q = query.lower()
    today = datetime.today().date()

    if "last 7 days" in q or "last seven days" in q:
        return (today - timedelta(days=7)).isoformat()

    if "last 30 days" in q or "last thirty days" in q:
        return (today - timedelta(days=30)).isoformat()

    if "this month" in q:
        return today.replace(day=1).isoformat()

    return None


def _where_clause(query: str, debit_only: bool = False) -> Tuple[str, Dict[str, Any]]:
    clauses = []
    params: Dict[str, Any] = {}

    tran_type = _extract_tran_type_from_query(query)
    start_date = _extract_start_date(query)

    if tran_type:
        clauses.append("upper(tran_type) = :tran_type")
        params["tran_type"] = tran_type

    if start_date:
        clauses.append("to_date(transaction_date) >= to_date(:start_date)")
        params["start_date"] = start_date

    if debit_only:
        clauses.append("lower(type) = 'debit'")

    if not clauses:
        return "", params

    return "WHERE " + " AND ".join(clauses), params


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0)
    except Exception:
        return 0.0


# ======================================================
# TRANSACTION TYPES
# ======================================================
def get_all_tran_types() -> List[str]:
    query = f"""
        SELECT DISTINCT tran_type
        FROM {DATABRICKS_TABLE}
        WHERE tran_type IS NOT NULL
        ORDER BY tran_type
    """

    rows = _run_sql(query)
    return [str(r["tran_type"]).strip() for r in rows if r.get("tran_type")]


def get_all_tran_types_with_count() -> List[Tuple[str, int]]:
    query = f"""
        SELECT tran_type, COUNT(*) AS txn_count
        FROM {DATABRICKS_TABLE}
        WHERE tran_type IS NOT NULL
        GROUP BY tran_type
        ORDER BY txn_count DESC, tran_type
    """

    rows = _run_sql(query)
    return [
        (str(r["tran_type"]).strip(), int(r["txn_count"]))
        for r in rows
        if r.get("tran_type")
    ]


# ======================================================
# VECTOR SEARCH TRANSACTIONS
# ======================================================
def search_transactions(query: str) -> List[Dict]:
    """
    Semantic transaction search using Databricks Vector Search.
    """
    try:
        raw = search_raw(query=query, limit=20)

        data_array = raw.get("result", {}).get("data_array", [])
        columns = raw.get("manifest", {}).get("columns", [])
        column_names = [c.get("name") for c in columns]

        results: List[Dict] = []

        for row in data_array:
            item = dict(zip(column_names, row))

            results.append({
                "date": item.get("transaction_date") or item.get("date"),
                "description": item.get("description"),
                "tran_type": item.get("tran_type"),
                "amount": item.get("amount"),
                "type": item.get("type"),
                "category": item.get("category"),
                "content": item.get("content"),
            })

        return results

    except Exception:
        logger.exception("Databricks Vector Search transaction search failed")
        return []


def search_transactions_hybrid(query: str) -> List[Dict]:
    return search_transactions(query)


# ======================================================
# CATEGORY SPEND
# ======================================================
def get_spending_by_category(query: str) -> List[Tuple[str, float]]:
    where_sql, params = _where_clause(query, debit_only=True)

    sql_query = f"""
        SELECT
            COALESCE(category, 'other') AS category,
            SUM(ABS(COALESCE(amount, 0))) AS total_spend
        FROM {DATABRICKS_TABLE}
        {where_sql}
        GROUP BY COALESCE(category, 'other')
        ORDER BY total_spend DESC
    """

    rows = _run_sql(sql_query, params)

    return [
        (str(r["category"]), _safe_float(r["total_spend"]))
        for r in rows
    ]


# ======================================================
# TOP EXPENSES / INSIGHTS
# ======================================================
def get_top_expenses(query: str) -> List[Dict]:
    where_sql, params = _where_clause(query, debit_only=True)

    sql_query = f"""
        SELECT
            transaction_date AS date,
            description,
            tran_type,
            amount,
            category,
            type
        FROM {DATABRICKS_TABLE}
        {where_sql}
        ORDER BY ABS(COALESCE(amount, 0)) DESC
        LIMIT 20
    """

    return _run_sql(sql_query, params)


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


# ======================================================
# MAIN QUERY HANDLER
# ======================================================
def handle_query(query: str) -> str:
    try:
        if not query or not query.strip():
            return "Please enter a valid banking question."

        q = query.lower()

        if "transaction type" in q and "count" in q:
            data = get_all_tran_types_with_count()
            if not data:
                return "No transaction data found"

            return "\n".join([f"{t}: {c}" for t, c in data])

        if "transaction type" in q:
            data = get_all_tran_types()
            if not data:
                return "No transaction types found"

            return ", ".join(data)

        if "spending" in q or "category" in q:
            data = get_spending_by_category(query)
            if not data:
                return "No spending data found"

            return "\n".join([f"{cat}: ₹{amt:,.2f}" for cat, amt in data])

        if "top expense" in q or "highest expense" in q:
            data = get_top_expenses(query)
            if not data:
                return "No expense data found"

            return "\n".join([
                f"{d.get('date')} | {d.get('description')} | ₹{abs(_safe_float(d.get('amount'))):,.2f}"
                for d in data[:10]
            ])

        if "insight" in q or "analysis" in q:
            data = get_financial_insights(query)

            top_category = data["top_category"]
            top_category_text = (
                f"{top_category[0]} - ₹{top_category[1]:,.2f}"
                if top_category
                else "Not available"
            )

            top_expenses_text = "\n".join([
                f"{e.get('description')} - ₹{abs(_safe_float(e.get('amount'))):,.2f}"
                for e in data["top_expenses"]
            ])

            return (
                f"Total Spend: ₹{data['total_spend']:,.2f}\n"
                f"Top Category: {top_category_text}\n\n"
                f"Top Expenses:\n{top_expenses_text}"
            )

        data = search_transactions(query)

        if not data:
            return "No transactions found"

        return "\n".join([
            f"{d.get('date')} | {d.get('description')} | {d.get('tran_type')} | ₹{_safe_float(d.get('amount')):,.2f}"
            for d in data[:10]
        ])

    except Exception as e:
        logger.exception("handle_query error")
        return f"Error processing request: {str(e)}"