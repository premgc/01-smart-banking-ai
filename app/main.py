import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from app.routes.query import router as query_router

app = FastAPI()
app.include_router(query_router)

from app.routes.banking import router as banking_router
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# ======================================================
# LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ======================================================
# APP INIT
# ======================================================
app = FastAPI(
    title="Smart Banking AI",
    version="1.0.0",
    description="AI-powered Smart Banking Assistant"
)

# ======================================================
# CORS
# ======================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in prod if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# ROUTES
# ======================================================
app.include_router(
    banking_router,
    prefix="/api/banking",   # 🔥 IMPORTANT FIX
    tags=["Smart Banking"]
)

# ======================================================
# ROOT
# ======================================================
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Smart Banking AI",
        "version": "1.0.0"
    }
