"""
Tree Re-Identification Service
ConvNeXt-Base with TTA for embedding extraction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Optional, List
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class GeM(nn.Module):
    """Generalized Mean Pooling."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)


class TreeReIdModel(nn.Module):
    """ConvNeXt-Base model for tree re-identification."""

    def __init__(self, backbone_name="convnext_base", embedding_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        backbone_dim = 1024  # convnext_base output dim
        self.pool = GeM()
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x, normalize=True):
        features = self.backbone(x)
        features = self.pool(features)
        features = features.view(features.size(0), -1)
        embeddings = self.embedding_head(features)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class ReIDService:
    """
    Tree Re-Identification service.
    Extracts embeddings with optional TTA for better accuracy.
    """
    _instance = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path or settings.reid_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.img_size = 224
        self.embedding_dim = settings.embedding_dim
        self._loaded = False

        # Base transform
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # TTA transforms
        self.tta_transforms = self._get_tta_transforms()

    def _get_tta_transforms(self):
        """TTA transforms for embedding extraction."""
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        size = self.img_size
        return [
            # Original
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                norm
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                norm
            ]),
            # Center crop (slightly zoomed)
            transforms.Compose([
                transforms.Resize((size + 32, size + 32)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                norm
            ]),
            # Center crop + flip
            transforms.Compose([
                transforms.Resize((size + 32, size + 32)),
                transforms.CenterCrop(size),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                norm
            ]),
        ]

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> bool:
        """Load re-ID model from checkpoint."""
        if self._loaded:
            return True

        if not self.model_path.exists():
            logger.warning(f"ReID model not found: {self.model_path}")
            return False

        logger.info(f"Loading ReID model from {self.model_path} on {self.device}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Get config from checkpoint
            config = checkpoint.get("config", {})
            backbone = config.get("backbone_name", "convnext_base")
            emb_dim = config.get("embedding_dim", 1024)

            self.model = TreeReIdModel(backbone_name=backbone, embedding_dim=emb_dim)

            # Load weights (handle classifier key if present)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            # Filter out classifier weights (not needed for inference)
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
            self.model.load_state_dict(state_dict, strict=False)

            self.model.to(self.device)
            self.model.eval()
            self._loaded = True
            logger.info(f"ReID model loaded: {backbone}, emb_dim={emb_dim}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ReID model: {e}")
            return False

    def extract_embedding(self, image: Image.Image, use_tta: bool = False) -> np.ndarray:
        """
        Extract embedding from image.

        Args:
            image: PIL Image
            use_tta: Use Test-Time Augmentation (slower but more accurate)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not self._loaded:
            if not self.load_model():
                # Return random embedding if model not loaded (for testing)
                logger.warning("Model not loaded, returning random embedding")
                return np.random.randn(self.embedding_dim).astype(np.float32)

        if image.mode != "RGB":
            image = image.convert("RGB")

        with torch.no_grad():
            if use_tta:
                embeddings = []
                for transform in self.tta_transforms:
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    emb = self.model(img_tensor).cpu().numpy()
                    embeddings.append(emb)
                # Average and re-normalize
                avg_emb = np.mean(embeddings, axis=0).squeeze()
                avg_emb = avg_emb / np.linalg.norm(avg_emb)
                return avg_emb.astype(np.float32)
            else:
                img_tensor = self.base_transform(image).unsqueeze(0).to(self.device)
                emb = self.model(img_tensor).cpu().numpy().squeeze()
                return emb.astype(np.float32)

    def extract_embedding_batch(self, images: List[Image.Image], use_tta: bool = False) -> np.ndarray:
        """Extract embeddings for batch of images."""
        return np.array([self.extract_embedding(img, use_tta=use_tta) for img in images])

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image (legacy compatibility)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image


# Global instance
_reid_service = None


def get_reid_service() -> ReIDService:
    """Dependency for FastAPI endpoints."""
    global _reid_service
    if _reid_service is None:
        _reid_service = ReIDService()
        _reid_service.load_model()
    return _reid_service
