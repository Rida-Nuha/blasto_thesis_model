"""
Multi-model evaluation for ICM classification (Seed 42 models)
Generates:
1. Confusion Matrix per model
2. ROC Curve overlay
3. Precision–Recall Curve overlay
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from PIL import Image
import os

# ============================================================
# CONFIGURATION (EDIT ONLY DATASET ROOT IF NEEDED)
# ============================================================
DATASET_ROOT = "/kaggle/input/dataset"   # ← must be your ONLY attached dataset
CSV_FILE = f"{DATASET_ROOT}/Gardner_train_silver.csv"
IMG_FOLDER = f"{DATASET_ROOT}/Images/Images"
TARGET_SCORE = "ICM_silver"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# ============================================================
# MODEL PATHS (FROM YOUR SCREENSHOT)
# ============================================================
MODELS = {
    "Swin-T": {
        "type": "swin",
        "ckpt": "/kaggle/working/fresh_clone/kaggle/working/saved_models/uncertainty_ICM/ICM_silver_seed42_best.pth",
        "input": 224
    },
    "ResNet-50": {
        "type": "resnet50",
        "ckpt": "/kaggle/working/fresh_clone/kaggle/working/saved_models/uncertainty_ICM_resnet50/ICM_resnet50_seed42_best.pth",
        "input": 224
    },
    "ResNet-18": {
        "type": "resnet18",
        "ckpt": "/kaggle/working/fresh_clone/kaggle/working/saved_models/uncertainty_ICM_resnet18/ICM_resnet18_seed42_best.pth",
        "input": 224
    },
    "DenseNet-121": {
        "type": "densenet121",
        "ckpt": "/kaggle/working/fresh_clone/kaggle/working/saved_models/uncertainty_ICM_densenet121/ICM_densenet121_seed42_best.pth",
        "input": 224
    },
    "Inception-V3": {
        "type": "inception",
        "ckpt": "/kaggle/working/fresh_clone/kaggle/working/saved_models/uncertainty_ICM_inceptionv3/ICM_inceptionv3_seed42_best.pth",
        "input": 299
    }
}

# ============================================================
# DATASET
# ============================================================
class EmbryoDataset(Dataset):
    def __init__(self, df, img_folder, transform):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(
            os.path.join(self.img_folder, self.df.loc[idx, "Image"])
        ).convert("RGB")
        label = self.df.loc[idx, "label"]
        return self.transform(img), torch.tensor(label, dtype=torch.long)

# ============================================================
# LOAD VALIDATION SPLIT (SAME AS TRAINING)
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

def load_model(model_type, ckpt):
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

    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.to(DEVICE).eval()

# ============================================================
# EVALUATION
# ============================================================
results = {}

for name, cfg in MODELS.items():
    print(f"Evaluating {name}...")

    transform = transforms.Compose([
        transforms.Resize((cfg["input"], cfg["input"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    loader = DataLoader(
        EmbryoDataset(val_df, IMG_FOLDER, transform),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = load_model(cfg["type"], cfg["ckpt"])

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
    y_pred = (y_prob >= 0.5).astype(int)

    results[name] = {
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    }

# ============================================================
# ENSEMBLE CONFUSION MATRIX (SOFT VOTING)
# ============================================================

# Stack probabilities from all models
all_probs = []
all_labels = None

for name, r in results.items():
    all_probs.append(r["y_prob"])
    all_labels = r["y_true"]   # same for all models

# Average probabilities across models
ensemble_probs = np.mean(np.stack(all_probs, axis=0), axis=0)

# Final predictions
ensemble_preds = (ensemble_probs >= 0.5).astype(int)

# Confusion matrix
cm = confusion_matrix(all_labels, ensemble_preds)

plt.figure(figsize=(4.5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Poor", "Good"],
    yticklabels=["Poor", "Good"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Ensemble Confusion Matrix (ICM)")
plt.tight_layout()
plt.show()

# ============================================================
# ROC OVERLAY
# ============================================================
plt.figure(figsize=(6, 5))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – ICM Models")
plt.legend()
plt.tight_layout()
plt.show()

# ============================================================
# PR OVERLAY
# ============================================================
plt.figure(figsize=(6, 5))
for name, r in results.items():
    precision, recall, _ = precision_recall_curve(r["y_true"], r["y_prob"])
    ap = average_precision_score(r["y_true"], r["y_prob"])
    plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – ICM Models")
plt.legend()
plt.tight_layout()
plt.show()
