from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from dotenv import load_dotenv

from api.errors import register_exception_handlers
from api.routers import get_file_store, get_session_store, router


logger = logging.getLogger("app")

# Auto-load project root .env for local/dev runs.
# Existing process env vars still take precedence (override=False).
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


def _parse_cors_origins() -> list[str]:
    """
    Parse CORS origins from env.
    Defaults cover local Vite/Tauri development.
    """
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        (
            "http://127.0.0.1:5173,"
            "http://localhost:5173,"
            "http://127.0.0.1:1420,"
            "http://localhost:1420,"
            "tauri://localhost,"
            "http://tauri.localhost,"
            "https://tauri.localhost"
        ),
    )
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins


def _redact_text(value: str) -> str:
    """Mask long text fields before logging to avoid sensitive-content leakage."""
    if not value:
        return value
    return f"<redacted len={len(value)}>"


def _redact_payload(payload: object) -> object:
    """Recursively redact sensitive fields in JSON payload logs."""
    sensitive_keys = {"disclosure_text", "oa_text", "text", "content"}
    if isinstance(payload, dict):
        redacted: dict[str, object] = {}
        for key, val in payload.items():
            if key in sensitive_keys and isinstance(val, str):
                redacted[key] = _redact_text(val)
            else:
                redacted[key] = _redact_payload(val)
        return redacted
    if isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    return payload


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    App lifecycle hooks.
    Runs periodic session TTL cleanup to align with security data-retention requirement.
    """
    stop_event = asyncio.Event()
    store = get_session_store()
    file_store = get_file_store()
    ttl_days = int(os.getenv("SESSION_TTL_DAYS", "30"))
    upload_ttl_hours = int(os.getenv("UPLOAD_TTL_HOURS", "24"))

    async def _cleanup_loop() -> None:
        while not stop_event.is_set():
            removed = store.cleanup_expired(ttl_days=ttl_days)
            if removed > 0:
                logger.info("event=session_cleanup removed=%s ttl_days=%s", removed, ttl_days)
            removed_files = file_store.cleanup_expired(ttl_hours=upload_ttl_hours)
            if removed_files > 0:
                logger.info("event=file_cleanup removed=%s ttl_hours=%s", removed_files, upload_ttl_hours)
            await asyncio.sleep(3600)

    task = asyncio.create_task(_cleanup_loop())
    try:
        yield
    finally:
        stop_event.set()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


def create_app() -> FastAPI:
    """Application factory wiring routers, middleware, and unified exception handlers."""
    app = FastAPI(title="M-Cube", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_parse_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_observability_middleware(request: Request, call_next):
        """
        Structured request logging middleware:
        - Adds request id header
        - Logs method/path/status/latency
        - Redacts sensitive text fields for JSON payload logging
        """
        request_id = request.headers.get("X-Request-Id", str(uuid4()))
        raw_body = await request.body()
        content_type = request.headers.get("content-type", "")
        if raw_body and "application/json" in content_type:
            try:
                payload = json.loads(raw_body.decode("utf-8"))
                logger.info("event=request_payload request_id=%s payload=%s", request_id, _redact_payload(payload))
            except Exception:  # noqa: BLE001
                logger.info("event=request_payload request_id=%s payload=<non-json-or-unparseable>", request_id)

        async def receive():
            return {"type": "http.request", "body": raw_body, "more_body": False}

        request = Request(request.scope, receive)
        start = time.perf_counter()
        response: Response = await call_next(request)
        latency_ms = int((time.perf_counter() - start) * 1000)
        response.headers["X-Request-Id"] = request_id
        logger.info(
            "event=http_request request_id=%s method=%s path=%s status=%s latency_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        return response

    app.include_router(router)
    register_exception_handlers(app)
    return app


app = create_app()
