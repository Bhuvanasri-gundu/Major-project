"""
backend/app.py — FastAPI Application Entrypoint.
"""

from pathlib import Path
import sys
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("BackendApp")

# Ensure project root is on sys.path and .env is loaded
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from backend.routes.api import router as api_router

app = FastAPI(
    title="Ontology-Guided Hypothesis Generation API",
    description="Evidence-grounded hypothesis generation in mHealth research",
    version="1.0.0",
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "message": "Ontology-Guided Hypothesis Generation API is active.",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
