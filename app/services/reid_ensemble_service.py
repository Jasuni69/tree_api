"""
Ensemble Tree Re-Identification Service
Combines Swin + ConvNeXt models for improved accuracy (~82% CMC@1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Optional, List, Tuple
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


class ConvNeXtReID(nn.Module):
    """ConvNeXt-Base model for tree re-identification."""

    def __init__(self, backbone_name="convnext_base", embedding_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="")
        backbone_dim = 1024
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


class SwinReID(nn.Module):
    """Swin-Base model for tree re-identification."""

    def __init__(self, backbone_name="swin_base_patch4_window7_224", embedding_dim=1024, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, global_pool="avg")
        backbone_dim = 1024
        self.embedding_head = nn.Sequential(
            nn.Linear(backbone_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_dim)
        )

    def forward(self, x, normalize=True):
        features = self.backbone(x)
        embeddings = self.embedding_head(features)
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class EnsembleReIDService:
    """
    Ensemble Re-ID service combining Swin + ConvNeXt.
    Weighted average of embeddings for better accuracy.
    """
    _instance = None

    def __init__(
        self,
        convnext_path: Optional[str] = None,
        swin_path: Optional[str] = None,
        weights: Tuple[float, float] = (0.5, 0.5)
    ):
        self.convnext_path = Path(convnext_path or settings.reid_model_path)
        self.swin_path = Path(swin_path or settings.swin_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.convnext_model = None
        self.swin_model = None
        self.weights = weights  # (swin_weight, convnext_weight)

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
            transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize((size + 32, size + 32)), transforms.CenterCrop(size), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize((size + 32, size + 32)), transforms.CenterCrop(size), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        ]

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_convnext(self) -> bool:
        """Load ConvNeXt model."""
        if not self.convnext_path.exists():
            logger.warning(f"ConvNeXt model not found: {self.convnext_path}")
            return False

        try:
            checkpoint = torch.load(self.convnext_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})
            backbone = config.get("backbone_name", "convnext_base")
            emb_dim = config.get("embedding_dim", 1024)

            self.convnext_model = ConvNeXtReID(backbone_name=backbone, embedding_dim=emb_dim)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
            self.convnext_model.load_state_dict(state_dict, strict=False)
            self.convnext_model.to(self.device)
            self.convnext_model.eval()
            logger.info(f"ConvNeXt loaded: {backbone}")
            return True
        except Exception as e:
            logger.error(f"Failed to load ConvNeXt: {e}")
            return False

    def _load_swin(self) -> bool:
        """Load Swin model."""
        if not self.swin_path.exists():
            logger.warning(f"Swin model not found: {self.swin_path}")
            return False

        try:
            checkpoint = torch.load(self.swin_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {})
            backbone = config.get("backbone_name", "swin_base_patch4_window7_224")
            emb_dim = config.get("embedding_dim", 1024)

            self.swin_model = SwinReID(backbone_name=backbone, embedding_dim=emb_dim)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
            self.swin_model.load_state_dict(state_dict, strict=False)
            self.swin_model.to(self.device)
            self.swin_model.eval()
            logger.info(f"Swin loaded: {backbone}")
            return True
        except Exception as e:
            logger.error(f"Failed to load Swin: {e}")
            return False

    def load_models(self) -> bool:
        """Load both ensemble models."""
        if self._loaded:
            return True

        logger.info(f"Loading ensemble models on {self.device}")

        convnext_ok = self._load_convnext()
        swin_ok = self._load_swin()

        if convnext_ok and swin_ok:
            self._loaded = True
            logger.info(f"Ensemble ready: weights swin={self.weights[0]}, convnext={self.weights[1]}")
            return True
        elif convnext_ok:
            # Fallback to ConvNeXt only
            self._loaded = True
            self.weights = (0.0, 1.0)
            logger.warning("Swin not loaded, using ConvNeXt only")
            return True
        elif swin_ok:
            # Fallback to Swin only
            self._loaded = True
            self.weights = (1.0, 0.0)
            logger.warning("ConvNeXt not loaded, using Swin only")
            return True

        return False

    def _extract_single(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Extract ensemble embedding from tensor."""
        w_swin, w_convnext = self.weights
        embeddings = []

        with torch.no_grad():
            if self.swin_model and w_swin > 0:
                swin_emb = self.swin_model(img_tensor)
                embeddings.append(w_swin * swin_emb)

            if self.convnext_model and w_convnext > 0:
                convnext_emb = self.convnext_model(img_tensor)
                embeddings.append(w_convnext * convnext_emb)

        combined = sum(embeddings)
        combined = F.normalize(combined, p=2, dim=1)
        return combined.cpu().numpy().squeeze()

    def extract_embedding(self, image: Image.Image, use_tta: bool = True) -> np.ndarray:
        """
        Extract ensemble embedding from image.

        Args:
            image: PIL Image
            use_tta: Use Test-Time Augmentation (default True for ensemble)

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not self._loaded:
            if not self.load_models():
                logger.warning("Models not loaded, returning random embedding")
                return np.random.randn(self.embedding_dim).astype(np.float32)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if use_tta:
            embeddings = []
            for transform in self.tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                emb = self._extract_single(img_tensor)
                embeddings.append(emb)
            avg_emb = np.mean(embeddings, axis=0)
            avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
            return avg_emb.astype(np.float32)
        else:
            img_tensor = self.base_transform(image).unsqueeze(0).to(self.device)
            return self._extract_single(img_tensor).astype(np.float32)

    def extract_embedding_batch(self, images: List[Image.Image], use_tta: bool = True) -> np.ndarray:
        """Extract embeddings for batch of images."""
        return np.array([self.extract_embedding(img, use_tta=use_tta) for img in images])

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image (legacy compatibility)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image


# Global instance
_ensemble_service = None


def get_ensemble_reid_service() -> EnsembleReIDService:
    """Dependency for FastAPI endpoints."""
    global _ensemble_service
    if _ensemble_service is None:
        _ensemble_service = EnsembleReIDService()
        _ensemble_service.load_models()
    return _ensemble_service
