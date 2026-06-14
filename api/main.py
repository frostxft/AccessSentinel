"""FastAPI main application for AccessSentinel."""

from __future__ import annotations

import logging
import os
import re
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

load_dotenv()

from api.routes import identity, report, risk  # noqa: E402

# ── Logger ────────────────────────────────────────────────────────────────────

logger = logging.getLogger("accesssentinel")
logger.setLevel(logging.DEBUG)

_json_handler = logging.StreamHandler()
_json_handler.setFormatter(
    logging.Formatter(
        '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    )
)
logger.handlers.clear()
logger.addHandler(_json_handler)
logger.propagate = False


# ── Masking helpers ────────────────────────────────────────────────────────────


def _mask_email(value: str) -> str:
    return re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "u***@***.com", value)


def _mask_username(value: str) -> str:
    return re.sub(r"(?<![@\w])([a-zA-Z]{2})[a-zA-Z0-9._-]*", r"\1***", value)


# ── Lifespan ───────────────────────────────────────────────────────────────────

models_loaded = False
baseline_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_loaded, baseline_loaded
    baseline_dir = os.path.join("data", "baselines")
    models_dir = os.path.join("models")

    if os.path.isdir(baseline_dir) and any(os.scandir(baseline_dir)):
        baseline_loaded = True
        logger.info("Baselines loaded from %s", baseline_dir)
    else:
        logger.warning("No baseline files found in %s", baseline_dir)

    if os.path.isdir(models_dir) and any(
        f.name.endswith(".pkl") for f in os.scandir(models_dir)
    ):
        models_loaded = True
        logger.info("Models loaded from %s", models_dir)
    else:
        logger.warning("No .pkl model files found in %s", models_dir)

    yield


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Access Sentinel API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Middleware (registered in order: CORS -> RequestID -> LogMasking) ──────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def log_masking_middleware(request: Request, call_next):
    original_make_record = logging.Logger.makeRecord

    def patched_make_record(self, name, level, fn, lno, msg, args, exc_info, func=None, extra=None, sinfo=None):
        msg = _mask_username(_mask_email(str(msg)))
        if args:
            args = tuple(
                _mask_username(_mask_email(str(a))) if isinstance(a, str) else a
                for a in args
            )
        return original_make_record(self, name, level, fn, lno, msg, args, exc_info, func=func, extra=extra, sinfo=sinfo)

    logging.Logger.makeRecord = patched_make_record
    try:
        response = await call_next(request)
    finally:
        logging.Logger.makeRecord = original_make_record

    return response


# ── Exception handler ──────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("Unhandled exception", extra={"request_id": request_id})
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "request_id": request_id},
    )


# ── Rate limiting ──────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


# ── Routes ─────────────────────────────────────────────────────────────────────

app.include_router(identity.router)
app.include_router(risk.router)
app.include_router(report.router)
