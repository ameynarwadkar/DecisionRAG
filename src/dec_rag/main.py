from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from dec_rag.config import Settings, get_settings
from dec_rag.database import AsyncSessionLocal, init_db
from dec_rag.routers import api_keys, documents, judges, namespaces, retrieval, runs
from dec_rag.schemas import HealthResponse
from dec_rag.services.demo_seed import ensure_demo_corpus
from dec_rag.services.generator import is_generator_configured, resolve_litellm_params
from dec_rag.services.judges import DecisionJudgeError, list_judge_statuses

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    if get_settings().SEED_DEMO_CORPUS:
        async with AsyncSessionLocal() as session:
            await ensure_demo_corpus(session)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="DecisionRAG",
        description="Retriever-free document navigation backend inspired by arXiv:2610.35868",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(DecisionJudgeError)
    async def handle_decision_judge_error(
        _request: Request,
        exc: DecisionJudgeError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.detail,
                "judge_id": exc.judge_id,
            },
        )

    @app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
    async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
        gen_model, _ = resolve_litellm_params(settings.GENERATOR_MODEL, settings)
        return HealthResponse(
            status="ok",
            default_judge=settings.DEFAULT_JUDGE,
            judges=list_judge_statuses(settings),
            generator_model=gen_model,
            generator_configured=is_generator_configured(settings),
        )

    app.include_router(namespaces.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(retrieval.router, prefix="/api/v1")
    app.include_router(judges.router, prefix="/api/v1")
    app.include_router(runs.router, prefix="/api/v1")
    app.include_router(api_keys.router, prefix="/api/v1")

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_playground() -> Response:
        index_file = STATIC_DIR / "index.html"
        if index_file.is_file():
            return FileResponse(index_file)
        return JSONResponse({"status": "ok", "docs": "/docs"})

    return app


app = create_app()