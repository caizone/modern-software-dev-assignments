"""
FastAPI application with proper lifecycle management.

Rationale:
- Lifespan context manager for startup/shutdown patterns
- Centralized exception handlers convert domain exceptions to HTTP responses
- Logging configuration at application startup
- Clean separation between initialization and request handling
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import init_db, reset_db_manager
from .exceptions import AppException, NotFoundError, ValidationError
from .routers import action_items, notes


# =============================================================================
# Logging Configuration
# =============================================================================


def configure_logging() -> None:
    """Configure application logging based on settings."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =============================================================================
# Application Lifecycle
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    
    Startup:
    - Configure logging
    - Initialize database schema
    
    Shutdown:
    - Clean up database connections
    """
    # Startup
    configure_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting application...")
    
    settings = get_settings()
    logger.info(f"Database path: {settings.db_path}")
    logger.info(f"LLM model: {settings.llm_model}")
    
    init_db()
    logger.info("Database initialized")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down application...")
    reset_db_manager()
    logger.info("Cleanup complete")


# =============================================================================
# Application Instance
# =============================================================================


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# Exception Handlers
# =============================================================================


@app.exception_handler(ValidationError)
async def validation_error_handler(
    request: Request,
    exc: ValidationError,
) -> JSONResponse:
    """Handle validation errors with 400 status."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(NotFoundError)
async def not_found_error_handler(
    request: Request,
    exc: NotFoundError,
) -> JSONResponse:
    """Handle not found errors with 404 status."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(AppException)
async def app_exception_handler(
    request: Request,
    exc: AppException,
) -> JSONResponse:
    """Handle all application exceptions with appropriate status codes."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.message, "code": exc.code},
    )


# =============================================================================
# Routes
# =============================================================================


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Serve the frontend HTML page."""
    html_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    return html_path.read_text(encoding="utf-8")


# Include routers
app.include_router(notes.router)
app.include_router(action_items.router)

# Mount static files
static_dir = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
