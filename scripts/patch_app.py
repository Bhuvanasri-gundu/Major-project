from pathlib import Path

app_file = Path(r"E:\Major_Project_Implementation\backend\app.py")
content = app_file.read_text(encoding="utf-8")

old_fastapi_block = """from backend.routes.api import router as api_router

app = FastAPI(
    title="Ontology-Guided Hypothesis Generation API",
    description="Evidence-grounded hypothesis generation in mHealth research",
    version="1.0.0",
)"""

new_fastapi_block = """from contextlib import asynccontextmanager
import asyncio
from backend.routes.api import router as api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing and pre-warming PipelineService models during application startup...")
    from backend.services.pipeline_service import get_pipeline_service
    # Run pre-warming in a worker thread so event loop remains responsive
    await asyncio.to_thread(get_pipeline_service)
    logger.info("PipelineService models successfully pre-warmed. Server ready for warm requests.")
    yield

app = FastAPI(
    title="Ontology-Guided Hypothesis Generation API",
    description="Evidence-grounded hypothesis generation in mHealth research",
    version="1.0.0",
    lifespan=lifespan,
)"""

assert old_fastapi_block in content, "Could not find old_fastapi_block"
content = content.replace(old_fastapi_block, new_fastapi_block, 1)

app_file.write_text(content, encoding="utf-8")
print("backend/app.py successfully patched with lifespan!")
