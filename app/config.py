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
    reid_model_path: str = "./models/reid/best_model.pth"  # ConvNeXt
    swin_model_path: str = "./models/reid/swin_model.pth"  # Swin (for ensemble)
    health_model_dir: str = "./models/health"

    # Ensemble config
    use_ensemble: bool = False  # ConvNeXt+TTA beats ensemble (80.07% vs 78.23%)
    ensemble_weights: tuple = (0.5, 0.5)  # (swin, convnext)

    # Embedding config
    embedding_dim: int = 1024  # ConvNeXt embedding dimension
    similarity_threshold: float = 0.7

    class Config:
        env_file = ".env"


settings = Settings()
