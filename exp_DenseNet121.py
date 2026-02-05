"""
DenseNet-121 with MC Dropout for Uncertainty Estimation
Correct & stable implementation (Kaggle-ready)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import densenet121, DenseNet121_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os, random

# ============================================================
# CONFIG
# ============================================================
ARCH = "densenet121"
TARGET_SCORE = "EXP_silver"

TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
SAVE_DIR = "kaggle/working/saved_models/uncertainty"

NUM_CLASSES = 2
BINARY_THRESHOLD = 2

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 5e-4
DROPOUT_RATE = 0.3
TRAIN_SPLIT = 0.85
PATIENCE = 15
NUM_WORKERS = 2

SEEDS = [42]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
class GardnerDataset(Dataset):
    def __init__(self, csv_file, img_folder, target, transform=None):
        df = pd.read_csv(csv_file, sep=';')
        df = df[df[target].notna() & ~df[target].isin(['ND', 'NA'])]
        df[target] = pd.to_numeric(df[target])
        df["label"] = (df[target] >= BINARY_THRESHOLD).astype(int)

        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_folder, row["Image"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label"]

    def get_class_weights(self):
        counts = self.df["label"].value_counts().sort_index().values
        total = len(self.df)
        return torch.FloatTensor(total / (len(counts) * counts))

# ============================================================
# TRANSFORMS (224×224)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# DENSENET-121 WITH MC DROPOUT
# ============================================================
class DenseNet121WithUncertainty(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()

        self.backbone = densenet121(
            weights=DenseNet121_Weights.IMAGENET1K_V1
        )

        in_features = self.backbone.classifier.in_features  # 1024

        self.backbone.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

# ============================================================
# LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ============================================================
# TRAIN / VALIDATE
# ============================================================
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return total_loss / total, 100 * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, 100 * correct / total

# ============================================================
# TRAIN SINGLE MODEL
# ============================================================
def train_single_model(seed, train_loader, val_loader, class_weights):
    print(f"\nTraining {ARCH} — seed {seed}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = DenseNet121WithUncertainty(NUM_CLASSES, DROPOUT_RATE).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0
    patience = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train: Loss={tr_loss:.4f}, Acc={tr_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/{ARCH}_{TARGET_SCORE}_seed{seed}_best.pth"
            )
            print("   🎯 Saved best model")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("   🛑 Early stopping")
                break
        print()

# ============================================================
# MAIN
# ============================================================
def main():
    dataset = GardnerDataset(TRAIN_CSV, IMG_FOLDER, TARGET_SCORE)
    train_len = int(TRAIN_SPLIT * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    class_weights = dataset.get_class_weights().to(DEVICE)

    for seed in SEEDS:
        train_single_model(seed, train_loader, val_loader, class_weights)

if __name__ == "__main__":
    main()
