"""FastAPI application: GET /health, POST /chat, GET /sessions/{id}/messages.

Run with:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Settings
from .db import Database
from .llm import AllProvidersUnavailable, LLMGateway
from .schemas import ChatRequest, ChatResponse, HealthResponse
from .service import ChatService
from .simulation import InvalidSimulationHeader, parse_simulation_header

logger = logging.getLogger("chatbot.api")


def create_app(settings: Settings | None = None, gateway: LLMGateway | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.database_path)
        db.init()  # schema is created automatically on startup
        app.state.db = db
        app.state.gateway = gateway or LLMGateway(settings)
        app.state.service = ChatService(settings, db, app.state.gateway)
        logger.info(
            "started: keys=%d cache=%s simulation=%s",
            len(settings.api_keys),
            settings.cache_enabled,
            settings.enable_failure_simulation,
        )
        yield

    # Exactly three endpoints: the interactive docs routes are switched off.
    app = FastAPI(
        title="LLM Chatbot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )

    allow_all = "*" in settings.allowed_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else list(settings.allowed_origins),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Last line of defence: never a stack trace or an HTML page.
        logger.error("unhandled error: %s", type(exc).__name__, exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> Any:
        if not request.app.state.db.ping():
            raise HTTPException(status_code=503, detail="database unavailable")
        if not settings.api_keys:
            raise HTTPException(status_code=503, detail="no API keys configured")
        return {
            "status": "ok",
            "keys_configured": len(settings.api_keys),
            "cache_enabled": settings.cache_enabled,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        body: ChatRequest,
        request: Request,
        x_simulate_failure: str | None = Header(default=None),
    ) -> Any:
        simulation: dict[str, int] = {}
        if settings.enable_failure_simulation:
            try:
                simulation = parse_simulation_header(x_simulate_failure)
            except InvalidSimulationHeader as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
        try:
            return await request.app.state.service.chat(
                body.session_id, body.message, body.system_prompt, simulation
            )
        except AllProvidersUnavailable:
            raise HTTPException(status_code=503, detail="all providers unavailable") from None

    @app.get("/sessions/{session_id}/messages")
    async def session_messages(session_id: str, request: Request) -> Any:
        messages = await request.app.state.service.history(session_id)
        if messages is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session_id": session_id, "messages": messages}

    return app
