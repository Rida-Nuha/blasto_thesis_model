"""
ICM (Inner Cell Mass) Quality Prediction with Uncertainty Quantification
DenseNet-121 + Deep Ensembles + MC Dropout
WITH REAL-TIME TRAIN / VAL ACCURACY LOGGING
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
import warnings
import shutil
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_SCORE = "ICM_silver"
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "kaggle/working/saved_models/uncertainty_ICM_densenet121"
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
SEEDS = [42]
PATIENCE = 20

os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

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
# LOAD DATA
# ============================================================
df = pd.read_csv(TRAIN_CSV, sep=";")
df["label"] = df[TARGET_SCORE].apply(lambda x: 1 if x >= 2 else 0)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"]
)

# Class weights
class_counts = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (2 * class_counts)

if min(class_counts) / max(class_counts) < 0.25:
    class_weights[class_counts.argmin()] *= 1.5
    print("⚠️ High imbalance detected – boosting minority class")

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ============================================================
# TRANSFORMS (STANDARD 224x224)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================================================
# MODEL — DENSENET 121
# ============================================================
class DenseNetEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss *= self.alpha[targets]
        return loss.mean()

# ============================================================
# TRAINING FUNCTION (REAL-TIME OVERFITTING MONITORING)
# ============================================================
def train_model(seed):
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL WITH SEED {seed}")
    print(f"{'='*70}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(
        EmbryoDataset(train_df, IMG_FOLDER, train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        EmbryoDataset(val_df, IMG_FOLDER, val_transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    model = DenseNetEmbryoClassifier().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.5)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for x, y in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = out.argmax(1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_acc = 100.0 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                preds = out.argmax(1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = 100.0 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # -------- LOGGING --------
        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # -------- CHECKPOINT --------
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/ICM_densenet121_seed{seed}_best.pth"
            )
            print("  ✅ Best model saved")
        else:
            patience_counter += 1
            print(f"  ⏳ No improvement | Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    print(f"\n✅ Best Validation Accuracy (Seed {seed}): {best_acc:.2f}%")
    return best_acc

# ============================================================
# TRAIN ALL SEEDS
# ============================================================
results = {}
for seed in SEEDS:
    results[seed] = train_model(seed)

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("FINAL ICM DENSENET-121 RESULTS")
print("="*70)

for s, a in results.items():
    print(f"Seed {s}: {a:.2f}%")

print(f"\nAverage Accuracy: {np.mean(list(results.values())):.2f}%")

# ============================================================
# ARCHIVE MODELS
# ============================================================
shutil.make_archive("ICM_densenet121_models", "zip", MODEL_DIR)
print("\n✅ Models archived: ICM_densenet121_models.zip")
