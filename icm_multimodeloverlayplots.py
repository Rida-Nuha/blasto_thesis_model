"""
Multi-model overlay evaluation for ICM classification
Generates:
- ROC curve overlay
- Precision–Recall curve overlay

Supports:
Swin-T, ResNet-18, ResNet-50, DenseNet-121, Inception-V3
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from PIL import Image
import os

# ============================================================
# CONFIGURATION — EDIT THIS ONLY
# ============================================================
CSV_FILE = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
TARGET_SCORE = "ICM_silver"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of models to compare
MODELS = {
    "Swin-T": {
        "type": "swin",
        "checkpoint": "saved_models/uncertainty_ICM/swin_seed42_best.pth",
        "input_size": 224
    },
    "ResNet-50": {
        "type": "resnet50",
        "checkpoint": "saved_models/uncertainty_ICM_resnet50/ICM_resnet50_seed42_best.pth",
        "input_size": 224
    },
    "ResNet-18": {
        "type": "resnet18",
        "checkpoint": "saved_models/uncertainty_ICM_resnet18/ICM_resnet18_seed42_best.pth",
        "input_size": 224
    },
    "DenseNet-121": {
        "type": "densenet121",
        "checkpoint": "saved_models/uncertainty_ICM_densenet121/ICM_densenet121_seed42_best.pth",
        "input_size": 224
    },
    "Inception-V3": {
        "type": "inception",
        "checkpoint": "saved_models/uncertainty_ICM_inceptionv3/ICM_inceptionv3_seed42_best.pth",
        "input_size": 299
    }
}

# ============================================================
# DATASET
# ============================================================
class EmbryoDataset(Dataset):
    def __init__(self, df, img_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "Image"]
        label = self.df.loc[idx, "label"]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# ============================================================
# LOAD DATA (SAME SPLIT AS TRAINING)
# ============================================================
df = pd.read_csv(CSV_FILE, sep=";")
df["label"] = df[TARGET_SCORE].apply(lambda x: 1 if x >= 2 else 0)

_, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"]
)

# ============================================================
# MODEL FACTORY
# ============================================================
from torchvision.models import (
    resnet18, resnet50, densenet121,
    inception_v3, swin_t,
    ResNet18_Weights, ResNet50_Weights,
    DenseNet121_Weights, Inception_V3_Weights,
    Swin_T_Weights
)

def build_model(model_type):
    if model_type == "resnet18":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif model_type == "resnet50":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif model_type == "densenet121":
        m = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    elif model_type == "inception":
        m = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=False
        )
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif model_type == "swin":
        m = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        m.head = nn.Linear(m.head.in_features, 2)
    else:
        raise ValueError("Unknown model type")
    return m.to(DEVICE).eval()

# ============================================================
# EVALUATE EACH MODEL
# ============================================================
results = {}

for name, cfg in MODELS.items():
    print(f"Evaluating {name}...")

    transform = transforms.Compose([
        transforms.Resize((cfg["input_size"], cfg["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    loader = DataLoader(
        EmbryoDataset(val_df, IMG_FOLDER, transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = build_model(cfg["type"])
    model.load_state_dict(torch.load(cfg["checkpoint"], map_location=DEVICE))

    y_true, y_prob = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            prob = torch.softmax(out, dim=1)[:, 1]
            y_prob.extend(prob.cpu().numpy())
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    results[name] = {
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "ap": ap
    }

# ============================================================
# ROC OVERLAY PLOT
# ============================================================
plt.figure(figsize=(6, 5))
for name, r in results.items():
    plt.plot(r["fpr"], r["tpr"], label=f"{name} (AUC={r['roc_auc']:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — ICM Classification")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PR OVERLAY PLOT
# ============================================================
plt.figure(figsize=(6, 5))
for name, r in results.items():
    plt.plot(r["recall"], r["precision"], label=f"{name} (AP={r['ap']:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — ICM Classification")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PRINT SUMMARY
# ============================================================
print("\nSummary Metrics")
print("=" * 50)
for name, r in results.items():
    print(f"{name:15s} | ROC AUC: {r['roc_auc']:.4f} | AP: {r['ap']:.4f}")
