"""
FastAPI Application for ECG Arrhythmia Detection
==================================================
Main application entry point.

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.database import init_db
from src.api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="ECG Arrhythmia Detection API",
        description=(
            "AI-powered ECG arrhythmia detection and classification. "
            "Analyzes ECG signals and classifies heartbeats into "
            "AAMI standard classes (N, S, V, F, Q)."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router, prefix="/api")

    @app.on_event("startup")
    async def startup():
        """Initialize database on startup."""
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialized.")

        # Pre-load model
        logger.info("Pre-loading model...")
        try:
            from src.inference.predict import get_classifier

            classifier = get_classifier("cnn_baseline")
            if classifier.is_loaded:
                logger.info("Model loaded successfully.")
            else:
                logger.warning("Model not loaded.")
        except Exception as e:
            logger.error("Failed to load model: %s", e)

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "name": "ECG Arrhythmia Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/health",
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
