from app.analytics import (
    total_deposit,
    total_withdrawal,
    expense_breakdown,
    daily_summary,
    financial_insights,
    filtered_summary,
)

TOOLS = {
    "total_deposit": total_deposit,
    "total_withdrawal": total_withdrawal,
    "expense_breakdown": expense_breakdown,
    "daily_summary": daily_summary,
    "financial_insights": financial_insights,
    "filtered_summary": filtered_summary,
}