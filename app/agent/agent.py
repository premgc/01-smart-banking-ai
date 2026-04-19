import logging
from typing import Any, Dict, List, Tuple

from app.services.banking_service import (
    get_all_tran_types,
    get_all_tran_types_with_count,
    search_transactions,
    search_transactions_hybrid,
    get_spending_by_category,
    get_financial_insights,
)

logger = logging.getLogger(__name__)


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_transactions(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "No matching transactions found."

    lines = []
    for r in results[:10]:
        date = _safe_str(r.get("date"))
        tx_type = _safe_str(r.get("type"))
        tran_type = _safe_str(r.get("tran_type"))
        amount = _safe_str(r.get("amount"))
        category = _safe_str(r.get("category"))
        description = _safe_str(r.get("description"))

        lines.append(
            f"Date: {date} | Type: {tx_type} | Tran Type: {tran_type} | "
            f"Amount: {amount} | Category: {category} | Description: {description}"
        )

    return "\n".join(lines)


def _format_tran_types(values: List[str]) -> str:
    if not values:
        return "No transaction types found."
    return "Transaction types: " + ", ".join(values)


def _format_tran_types_with_count(values: List[Tuple[str, int]]) -> str:
    if not values:
        return "No transaction types found."

    lines = ["Transaction types with count:"]
    for tran_type, count in values:
        lines.append(f"{tran_type}: {count}")

    return "\n".join(lines)


def _format_spending_by_category(values: List[Tuple[str, float]]) -> str:
    if not values:
        return "No spending data found."

    lines = ["Spending by category:"]
    for category, amount in values[:10]:
        lines.append(f"{category}: ₹{amount:,.2f}")

    return "\n".join(lines)


def _format_insights(insights: Dict[str, Any]) -> str:
    total_spend = insights.get("total_spend", 0.0)
    top_category = insights.get("top_category")
    top_expenses = insights.get("top_expenses", [])
    category_spend = insights.get("category_spend", [])

    lines = [f"Total spend: ₹{total_spend:,.2f}"]

    if top_category:
        lines.append(f"Top category: {top_category[0]} → ₹{top_category[1]:,.2f}")

    if category_spend:
        lines.append("\nTop categories:")
        for category, amount in category_spend:
            lines.append(f"- {category}: ₹{amount:,.2f}")

    if top_expenses:
        lines.append("\nTop expenses:")
        for item in top_expenses:
            lines.append(
                f"- {item.get('date', '')} | {item.get('description', '')} | "
                f"₹{abs(float(item.get('amount') or 0)):,.2f}"
            )

    return "\n".join(lines)


def run_agent(query: str) -> str:
    try:
        q = query.lower().strip()
        logger.info("Processing query: %s", query)

        # 1. Unique transaction types
        if (
            "transaction type" in q
            or "transaction types" in q
            or "tran type" in q
            or "distinct transaction type" in q
            or "unique transaction type" in q
        ):
            if "count" in q or "how many" in q:
                return _format_tran_types_with_count(get_all_tran_types_with_count())
            return _format_tran_types(get_all_tran_types())

        # 2. Spending by category
        if "spending by category" in q or "spend by category" in q or "category breakdown" in q:
            return _format_spending_by_category(get_spending_by_category(query))

        # 3. Insights / where am i losing money
        if (
            "where am i losing money" in q
            or "where am i spending" in q
            or "financial insights" in q
            or "insights" in q
            or "top expenses" in q
        ):
            return _format_insights(get_financial_insights(query))

        # 4. Hybrid retrieval path
        if "semantic" in q or "hybrid" in q or "vector" in q:
            return _format_transactions(search_transactions_hybrid(query))

        # 5. Transaction queries
        if any(word in q for word in ["upi", "neft", "imps", "atm", "cash", "transaction", "transactions"]):
            return _format_transactions(search_transactions(query))

        if "balance" in q:
            return "Balance feature coming soon."

        if "summary" in q:
            return "Summary feature coming soon."

        return "Sorry, I didn’t understand your request."

    except Exception as e:
        logger.exception("Agent error")
        return f"Agent error: {str(e)}"