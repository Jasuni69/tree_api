"""
Tree Health Assessment Service
Uses BeastModePredictor (EfficientNet-B3 ensemble + TTA)
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, List, Union
import logging

from app.config import settings

logger = logging.getLogger(__name__)

HEALTH_LABELS = ["irrigation_bag", "tree_status", "tree_supports", "support_ropes", "tree_base"]

DEFAULT_THRESHOLDS = {
    "irrigation_bag": 0.5,
    "tree_status": 0.5,
    "tree_supports": 0.5,
    "support_ropes": 0.55,
    "tree_base": 0.60,
}


class BeastModeClassifier(nn.Module):

    def __init__(self, num_labels=5, backbone="efficientnet_b3", dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0, drop_rate=dropout)
        num_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.25),
            nn.Linear(256, num_labels)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


def get_tta_transforms(img_size=300):
    base_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return [
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomHorizontalFlip(p=1.0), transforms.RandomVerticalFlip(p=1.0), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size + 20, img_size + 20)), transforms.CenterCrop(img_size), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size + 20, img_size + 20)), transforms.CenterCrop(img_size), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomRotation((5, 5)), transforms.ToTensor(), base_norm]),
        transforms.Compose([transforms.Resize((img_size, img_size)), transforms.RandomRotation((-5, -5)), transforms.ToTensor(), base_norm]),
    ]


class HealthService:
    _instance = None

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or settings.health_model_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = []
        self.thresholds = DEFAULT_THRESHOLDS
        self.img_size = 300
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.tta_transforms = get_tta_transforms(self.img_size)
        self._loaded = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        if self._loaded:
            return True
        model_files = list(self.model_dir.glob("beast_*.pth"))
        if not model_files:
            logger.warning(f"No health models found in {self.model_dir}")
            return False
        logger.info(f"Loading {len(model_files)} health models on {self.device}")
        for model_path in model_files:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                backbone = checkpoint.get("backbone", "efficientnet_b3")
                model = BeastModeClassifier(num_labels=5, backbone=backbone, dropout=0.4)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.to(self.device)
                model.eval()
                self.models.append(model)
                logger.info(f"Loaded: {model_path.name}")
            except Exception as e:
                logger.error(f"Failed to load {model_path}: {e}")
        self._loaded = len(self.models) > 0
        logger.info(f"Health service ready with {len(self.models)} models")
        return self._loaded

    def _predict_ensemble(self, img_tensor):
        all_probs = []
        with torch.no_grad():
            for model in self.models:
                with autocast():
                    output = model(img_tensor)
                probs = torch.sigmoid(output).cpu().numpy()
                all_probs.append(probs)
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs.squeeze()

    def _predict_with_tta(self, image):
        all_probs = []
        with torch.no_grad():
            for transform in self.tta_transforms:
                img_tensor = transform(image).unsqueeze(0).to(self.device)
                probs = self._predict_ensemble(img_tensor)
                all_probs.append(probs)
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def assess_health(self, image, use_tta=True):
        if not self._loaded:
            if not self.load_models():
                return {"error": "Health models not loaded", "probabilities": {}, "predictions": {}, "confidence": {}}
        if image.mode != "RGB":
            image = image.convert("RGB")
        if use_tta:
            probs = self._predict_with_tta(image)
        else:
            img_tensor = self.base_transform(image).unsqueeze(0).to(self.device)
            probs = self._predict_ensemble(img_tensor)
        result = {"probabilities": {}, "predictions": {}, "confidence": {}}
        for i, label in enumerate(HEALTH_LABELS):
            prob = float(probs[i])
            threshold = self.thresholds.get(label, 0.5)
            pred = prob >= threshold
            result["probabilities"][label] = round(prob, 4)
            result["predictions"][label] = pred
            result["confidence"][label] = round(prob if pred else (1 - prob), 4)
        result["overall_confidence"] = round(min(result["confidence"].values()), 4)
        return result

    def assess_health_batch(self, images, use_tta=True):
        return [self.assess_health(img, use_tta=use_tta) for img in images]


_health_service = None


def get_health_service():
    global _health_service
    if _health_service is None:
        _health_service = HealthService()
        _health_service.load_models()
    return _health_service