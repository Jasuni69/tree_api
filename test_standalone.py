"""
Standalone model test - no app imports needed.
Tests ReID and Health models directly.

Usage: python test_standalone.py [image_path]
"""
import sys
import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision import transforms
import timm
from PIL import Image
import numpy as np
from pathlib import Path

print("="*60)
print("Tree Identification - Standalone Model Test")
print("="*60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Get test image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
    print(f"Loading: {image_path}")
    image = Image.open(image_path).convert('RGB')
else:
    print("No image provided, creating random test image...")
    arr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    image = Image.fromarray(arr)

print(f"Image size: {image.size}")

# ============================================================
# ReID Model
# ============================================================
print("\n" + "-"*60)
print("REID MODEL (ConvNeXt-Base)")
print("-"*60)

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return torch.nn.functional.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1))
        ).pow(1. / self.p)

class TreeReIdModel(nn.Module):
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
    def forward(self, x):
        features = self.backbone(x)
        features = self.pool(features).view(features.size(0), -1)
        embeddings = self.embedding_head(features)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

reid_path = Path("models/reid/best_model.pth")
if reid_path.exists():
    print(f"Loading from {reid_path}...")
    start = time.time()
    checkpoint = torch.load(reid_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    reid_model = TreeReIdModel(
        backbone_name=config.get("backbone_name", "convnext_base"),
        embedding_dim=config.get("embedding_dim", 1024)
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k: v for k, v in state_dict.items() if "classifier" not in k}
    reid_model.load_state_dict(state_dict, strict=False)
    reid_model.to(device).eval()
    print(f"[OK] Loaded in {time.time()-start:.2f}s")

    # Transforms
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    base_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    tta_tfs = [
        transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize((256, 256)), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), norm]),
    ]

    # Test no TTA
    with torch.no_grad():
        start = time.time()
        tensor = base_tf(image).unsqueeze(0).to(device)
        emb = reid_model(tensor).cpu().numpy().squeeze()
        no_tta_time = time.time() - start
    print(f"[OK] No TTA: {no_tta_time*1000:.1f}ms, shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")

    # Test with TTA
    with torch.no_grad():
        start = time.time()
        embs = []
        for tf in tta_tfs:
            tensor = tf(image).unsqueeze(0).to(device)
            embs.append(reid_model(tensor).cpu().numpy())
        emb_tta = np.mean(embs, axis=0).squeeze()
        emb_tta = emb_tta / np.linalg.norm(emb_tta)
        tta_time = time.time() - start
    print(f"[OK] With TTA: {tta_time*1000:.1f}ms ({tta_time/no_tta_time:.1f}x slower)")
else:
    print(f"[FAIL] Model not found: {reid_path}")

# ============================================================
# Health Model
# ============================================================
print("\n" + "-"*60)
print("HEALTH MODEL (EfficientNet-B3 Ensemble)")
print("-"*60)

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
        return self.head(self.backbone(x))

health_dir = Path("models/health")
health_models = list(health_dir.glob("beast_*.pth")) if health_dir.exists() else []

if health_models:
    print(f"Found {len(health_models)} models")
    models = []
    start = time.time()
    for mp in health_models:
        checkpoint = torch.load(mp, map_location=device, weights_only=False)
        model = BeastModeClassifier(num_labels=5, backbone="efficientnet_b3")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        models.append(model)
        print(f"  [OK] {mp.name}")
    print(f"Loaded in {time.time()-start:.2f}s")

    LABELS = ["irrigation_bag", "tree_status", "tree_supports", "support_ropes", "tree_base"]
    health_tf = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Test
    with torch.no_grad():
        start = time.time()
        tensor = health_tf(image).unsqueeze(0).to(device)
        all_probs = []
        for m in models:
            with autocast():
                out = m(tensor)
            all_probs.append(torch.sigmoid(out).cpu().numpy())
        avg_probs = np.mean(all_probs, axis=0).squeeze()
        health_time = time.time() - start

    print(f"\n[OK] Inference: {health_time*1000:.1f}ms")
    print("\nPredictions:")
    for i, label in enumerate(LABELS):
        prob = avg_probs[i]
        pred = "YES" if prob >= 0.5 else "NO"
        print(f"  {label}: {pred} ({prob:.3f})")
else:
    print(f"[FAIL] No health models found in {health_dir}")

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
print("\nTo test with real image:")
print("  python test_standalone.py path/to/tree_image.jpg")
