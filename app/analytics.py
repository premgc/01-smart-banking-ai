import pandas as pd
import os
from datetime import datetime, timedelta

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "transactions.csv")


# =========================================================
# LOAD DATA (SAFE)
# =========================================================
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)

        # Ensure correct types
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        return df.dropna()

    except Exception as e:
        return pd.DataFrame()


# =========================================================
# TOTAL DEPOSIT
# =========================================================
def total_deposit():
    df = load_data()
    if df.empty:
        return "No data available"

    deposits = df[df["type"].str.lower() == "credit"]["amount"].sum()
    return f"Total deposits: £{deposits:,.2f}"


# =========================================================
# TOTAL WITHDRAWAL
# =========================================================
def total_withdrawal():
    df = load_data()
    if df.empty:
        return "No data available"

    withdrawals = df[df["type"].str.lower() == "debit"]["amount"].sum()
    return f"Total withdrawals: £{withdrawals:,.2f}"


# =========================================================
# EXPENSE BREAKDOWN
# =========================================================
def expense_breakdown():
    df = load_data()
    if df.empty:
        return "No data available"

    df = df[df["type"].str.lower() == "debit"]

    breakdown = df.groupby("category")["amount"].sum().sort_values(ascending=False)

    result = "\n".join([f"{cat}: £{amt:,.2f}" for cat, amt in breakdown.items()])

    return f"Expense breakdown:\n{result}"


# =========================================================
# DAILY SUMMARY
# =========================================================
def daily_summary():
    df = load_data()
    if df.empty:
        return "No data available"

    today = df["date"].max()

    day_df = df[df["date"] == today]

    income = day_df[day_df["type"] == "credit"]["amount"].sum()
    expense = day_df[day_df["type"] == "debit"]["amount"].sum()

    return (
        f"Daily Summary ({today.date()}):\n"
        f"Income: £{income:,.2f}\n"
        f"Expense: £{expense:,.2f}\n"
        f"Net: £{income - expense:,.2f}"
    )


# =========================================================
# FILTERED SUMMARY (DATE LOGIC)
# =========================================================
def filtered_summary(query: str):
    df = load_data()
    if df.empty:
        return "No data available"

    query = query.lower()

    today = datetime.today()

    if "last 7" in query:
        start = today - timedelta(days=7)

    elif "last 30" in query:
        start = today - timedelta(days=30)

    elif "this month" in query:
        start = today.replace(day=1)

    else:
        return "Could not understand date range"

    filtered = df[df["date"] >= start]

    income = filtered[filtered["type"] == "credit"]["amount"].sum()
    expense = filtered[filtered["type"] == "debit"]["amount"].sum()

    return (
        f"Filtered Summary:\n"
        f"Income: £{income:,.2f}\n"
        f"Expense: £{expense:,.2f}\n"
        f"Net: £{income - expense:,.2f}"
    )


# =========================================================
# FINANCIAL INSIGHTS
# =========================================================
def financial_insights():
    df = load_data()
    if df.empty:
        return "No data available"

    total_income = df[df["type"] == "credit"]["amount"].sum()
    total_expense = df[df["type"] == "debit"]["amount"].sum()

    savings = total_income - total_expense

    if savings < 0:
        return "⚠️ You are spending more than you earn. Reduce expenses."

    if savings < total_income * 0.2:
        return "⚠️ Low savings rate. Try to cut unnecessary expenses."

    return "✅ Healthy financial status. Keep it up!"