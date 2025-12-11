from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time

from app.config import settings
from app.database import init_db
from app.routers import trees, photos, identify
from app.services import get_reid_service, get_health_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await init_db()

    # Warm up models (load into GPU memory)
    logger.info("Warming up models...")
    start = time.time()

    try:
        reid_service = get_reid_service()
        if reid_service.load_model():
            logger.info(f"ReID model loaded in {time.time() - start:.2f}s")
        else:
            logger.warning("ReID model not available")
    except Exception as e:
        logger.error(f"Failed to load ReID model: {e}")

    try:
        health_service = get_health_service()
        # Health service loads on init
        logger.info("Health model loaded")
    except Exception as e:
        logger.error(f"Failed to load health model: {e}")

    logger.info(f"Model warm-up complete in {time.time() - start:.2f}s")

    yield
    # Shutdown: cleanup if needed


app = FastAPI(
    title="Tree Identification & Health Assessment API",
    description="Combined API for tree re-identification and health assessment",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(identify.router)
app.include_router(trees.router)
app.include_router(photos.router)


@app.get("/api/v1/health")
async def health_check():
    """API health check endpoint"""
    return {
        "status": "healthy",
        "service": "tree-identification-api",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Tree Identification & Health Assessment API",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
