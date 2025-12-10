from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Database - PostgreSQL with pgvector
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/tree_gallery"

    # Storage
    upload_dir: str = "./uploads"

    # CORS - allow mobile app
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8081", "*"]

    # Model paths (centralized in backend/models/)
    reid_model_path: str = "./models/reid/best_model.pth"
    health_model_dir: str = "./models/health"

    # Embedding config
    embedding_dim: int = 1024  # ConvNeXt embedding dimension
    similarity_threshold: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()
