from fastapi import APIRouter
from app.services.banking_service import handle_query

router = APIRouter()

@router.post("/query")
def query(data: dict):
    return {"response": handle_query(data.get("query"))}