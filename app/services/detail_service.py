"""
Detail Classification Service
Multi-head model for conditional detail labels:
- irrigation_bag_fill: 5-class (when irrigation_bag=True)
- tree_status_issue: 5-class (when tree_status=False)
- tree_base_issue: 3-class (when tree_base=False)
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
import timm
from PIL import Image
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Detail head configuration
DETAIL_HEADS = {
    'irrigation_bag_fill': {
        'num_classes': 5,
        'classes': ['100% Full', '75% Full', '50% Full', '25% Full', '0% Empty'],
        'parent': 'irrigation_bag',
        'parent_value': True
    },
    'tree_status_issue': {
        'num_classes': 5,
        'classes': ['Dry', 'Damaged', 'Diseased', 'Winter', 'Other'],
        'parent': 'tree_status',
        'parent_value': False
    },
    'tree_base_issue': {
        'num_classes': 3,
        'classes': ['Weeds', 'Debris', 'Trash'],
        'parent': 'tree_base',
        'parent_value': False
    }
}


class DetailClassifier(nn.Module):
    """Multi-head classifier for detail labels."""

    def __init__(self, backbone='efficientnet_b3', dropout=0.3):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
            drop_rate=dropout
        )

        num_features = self.backbone.num_features

        # Shared feature refinement
        self.shared_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5)
        )

        # Separate heads for each detail label
        self.heads = nn.ModuleDict()
        for head_name, head_info in DETAIL_HEADS.items():
            self.heads[head_name] = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.25),
                nn.Linear(128, head_info['num_classes'])
            )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_fc(features)

        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(shared)

        return outputs


class DetailService:
    """Service for detail classification inference."""
    _instance = None

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else Path(settings.detail_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.img_size = 300
        self._loaded = False

        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # TTA transforms
        self.tta_transforms = self._get_tta_transforms()

    def _get_tta_transforms(self):
        norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        size = self.img_size
        return [
            transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize((size, size)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize((size + 20, size + 20)), transforms.CenterCrop(size), transforms.ToTensor(), norm]),
            transforms.Compose([transforms.Resize((size + 20, size + 20)), transforms.CenterCrop(size), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        ]

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> bool:
        if self._loaded:
            return True

        if not self.model_path.exists():
            logger.warning(f"Detail model not found: {self.model_path}")
            return False

        logger.info(f"Loading detail model from {self.model_path} on {self.device}")

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

            backbone = checkpoint.get('backbone', 'efficientnet_b3')
            self.model = DetailClassifier(backbone=backbone)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info(f"Detail model loaded: {backbone}")
            return True

        except Exception as e:
            logger.error(f"Failed to load detail model: {e}")
            return False

    def assess_details(self, image: Image.Image, health_predictions: Dict[str, bool], use_tta: bool = True) -> Dict[str, Any]:
        """
        Assess detail labels based on health predictions.

        Args:
            image: PIL Image
            health_predictions: Dict of primary health predictions (e.g., {'irrigation_bag': True, 'tree_status': False})
            use_tta: Use test-time augmentation

        Returns:
            Dict with detail predictions for applicable labels
        """
        if not self._loaded:
            if not self.load_model():
                return {"error": "Detail model not loaded", "details": {}}

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get model outputs
        with torch.no_grad():
            if use_tta:
                all_outputs = {name: [] for name in DETAIL_HEADS.keys()}
                for transform in self.tta_transforms:
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                    with autocast():
                        outputs = self.model(img_tensor)
                    for name in DETAIL_HEADS.keys():
                        probs = torch.softmax(outputs[name], dim=1).cpu().numpy()
                        all_outputs[name].append(probs)

                # Average TTA predictions
                avg_outputs = {}
                for name in DETAIL_HEADS.keys():
                    avg_outputs[name] = np.mean(all_outputs[name], axis=0).squeeze()
            else:
                img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with autocast():
                    outputs = self.model(img_tensor)
                avg_outputs = {}
                for name in DETAIL_HEADS.keys():
                    avg_outputs[name] = torch.softmax(outputs[name], dim=1).cpu().numpy().squeeze()

        # Build result based on health predictions
        result = {"details": {}}

        for head_name, head_info in DETAIL_HEADS.items():
            parent_key = head_info['parent']
            parent_value = head_info['parent_value']

            # Only include if parent condition matches
            if parent_key in health_predictions and health_predictions[parent_key] == parent_value:
                probs = avg_outputs[head_name]
                pred_idx = int(np.argmax(probs))
                pred_class = head_info['classes'][pred_idx]
                confidence = float(probs[pred_idx])

                result["details"][head_name] = {
                    "prediction": pred_class,
                    "confidence": round(confidence, 4),
                    "all_probabilities": {
                        cls: round(float(probs[i]), 4)
                        for i, cls in enumerate(head_info['classes'])
                    }
                }

        return result


# Global instance
_detail_service = None


def get_detail_service() -> DetailService:
    global _detail_service
    if _detail_service is None:
        _detail_service = DetailService()
        _detail_service.load_model()
    return _detail_service
