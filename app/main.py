"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse

from app.agent.orchestrator import CommunicationAgent
from app.api.routes import router
from app.config import get_settings
from app.llm.factory import build_provider
from app.memory.store import SessionStore
from app.observability.logging_config import configure_logging
from app.observability.metrics import metrics
from app.observability.middleware import RequestTrackingMiddleware
from app.rag.knowledge import get_knowledge_base
from app.schemas import HealthResponse
from app.tools import build_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(
        level=settings.log_level,
        json_logs=settings.log_json,
        log_file=settings.log_file,
    )

    llm = build_provider(settings)
    registry = build_registry()
    memory = SessionStore(
        max_turns=settings.memory_max_turns, ttl_seconds=settings.session_ttl_seconds
    )

    app.state.settings = settings
    app.state.llm = llm
    app.state.registry = registry
    app.state.memory = memory
    app.state.agent = CommunicationAgent(
        llm=llm, registry=registry, memory=memory, settings=settings
    )

    if settings.enable_rag:
        get_knowledge_base()

    logger.info(
        "%s v%s started | provider=%s | tools=%s",
        settings.app_name,
        settings.version,
        llm.name,
        len(registry.names()),
    )
    try:
        yield
    finally:
        await llm.aclose()
        logger.info("shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "An agentic AI communication coach. Every request runs the pipeline: "
            "intent detection -> planning -> tool selection -> execution -> "
            "feedback synthesis, with short-term conversational memory."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        RequestTrackingMiddleware, slow_request_ms=settings.slow_request_ms
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Trace-Id", "X-Response-Time-ms"],
    )

    app.include_router(router, prefix=settings.api_prefix, tags=["coaching"])

    @app.get("/", tags=["meta"], summary="Service info")
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "api_prefix": settings.api_prefix,
            "endpoints": {
                "coaching_response": f"{settings.api_prefix}/coach",
                "communication_analysis": f"{settings.api_prefix}/analyze",
                "communication_improvement": f"{settings.api_prefix}/improve",
                "chat_history": f"{settings.api_prefix}/history/{{session_id}}",
                "resume_analysis": f"{settings.api_prefix}/resume/analyze",
                "interview_start": f"{settings.api_prefix}/interview/start",
                "interview_answer": f"{settings.api_prefix}/interview/answer",
                "knowledge_search": f"{settings.api_prefix}/knowledge/search",
            },
        }

    @app.get("/health", tags=["meta"], response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        llm = request.app.state.llm
        return HealthResponse(
            status="ok",
            version=settings.version,
            provider=llm.name,
            llm_available=llm.supports_generation and llm.available,
            uptime_seconds=metrics.uptime_seconds,
        )

    @app.get("/metrics", tags=["meta"], summary="Metrics snapshot (JSON)")
    async def metrics_json() -> dict:
        return metrics.snapshot()

    @app.get(
        "/metrics/prometheus",
        tags=["meta"],
        summary="Metrics in Prometheus text format",
        response_class=PlainTextResponse,
    )
    async def metrics_prometheus() -> str:
        return metrics.prometheus()

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", "-")
        logger.warning(
            "http error %s on %s: %s", exc.status_code, request.url.path, exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": f"http_{exc.status_code}",
                "detail": str(exc.detail),
                "trace_id": trace_id,
            },
            headers={"X-Trace-Id": trace_id},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        trace_id = getattr(request.state, "trace_id", "-")
        logger.warning("validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": exc.errors(),
                "trace_id": trace_id,
            },
            headers={"X-Trace-Id": trace_id},
        )

    return app


app = create_app()
