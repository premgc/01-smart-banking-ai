from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import logging

from app.agent import run_agent   # ✅ REAL LOGIC
import os

def handle_query(query: str):
    query = query.lower()

    if "transaction type" in query:
        return "Transaction types: ATM, UPI, NEFT, IMPS"

    if "spending" in query:
        return "Spending: ₹50000"

    return "I don’t understand the query"


# ======================================================
# ROUTER
# ======================================================
router = APIRouter()
logger = logging.getLogger(__name__)


# ======================================================
# REQUEST MODEL (VALIDATED)
# ======================================================
class BankingRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


# ======================================================
# RESPONSE MODEL
# ======================================================
class BankingResponse(BaseModel):
    success: bool
    reply: str


# ======================================================
# HEALTH CHECK
# ======================================================
@router.get("/health", status_code=status.HTTP_200_OK)
def health():
    return {
        "status": "ok",
        "service": "smart-banking-ai"
    }


# ======================================================
# MAIN ENDPOINT (PRODUCTION)
# ======================================================
@router.post(
    "/ask",
    response_model=BankingResponse,
    status_code=status.HTTP_200_OK
)
def ask_banking(payload: BankingRequest):
    try:
        question = payload.question.strip()

        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="question is required"
            )

        logger.info(f"Incoming banking query: {question}")

        # ======================================================
        # CALL AGENT (REAL BUSINESS LOGIC)
        # ======================================================
        answer = run_agent(question)

        if not answer:
            logger.warning("Empty response from agent")
            return BankingResponse(
                success=False,
                reply="No response from banking system."
            )

        return BankingResponse(
            success=True,
            reply=str(answer)
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Banking API failure")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Banking API error: {str(e)}"
        )