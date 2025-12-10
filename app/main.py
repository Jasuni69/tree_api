from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings
from app.database import init_db
from app.routers import trees, photos, identify


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await init_db()
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
