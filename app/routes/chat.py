from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.llm_service import LLMConfigError, ask_llm

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.get("/health")
def health():
    return {"status": "ok", "service": "smart-banking-ai"}

@router.post("/ask")
def ask(req: ChatRequest):
    try:
        result = ask_llm(req.question)
        if isinstance(result, dict):
            return result
        return {"reply": str(result)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except LLMConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")