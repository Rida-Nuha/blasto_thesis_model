"""
ResNet-18 with MC Dropout for Uncertainty Estimation
Hybrid uncertainty quantification for embryo quality prediction
KAGGLE VERSION - All paths fixed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import random
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_SCORE = "EXP_silver"

TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
SAVE_DIR = "/kaggle/working/saved_models/uncertainty"

BINARY_THRESHOLD = 2

NUM_CLASSES = 2
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.85
NUM_WORKERS = 2
PATIENCE = 15

SEEDS = [42]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
class GardnerDataset(Dataset):
    def __init__(self, csv_file, img_folder, target_column, threshold=2, transform=None):
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.target_column = target_column
        self.threshold = threshold
        self.transform = transform

        valid = (
            self.df[target_column].notna() &
            (self.df[target_column] != 'ND') &
            (self.df[target_column] != 'NA')
        )
        self.df = self.df[valid].copy()
        self.df[target_column] = pd.to_numeric(self.df[target_column], errors='coerce')
        self.df = self.df[self.df[target_column].notna()].copy()
        self.df['binary_label'] = (self.df[target_column] >= threshold).astype(int)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Image'])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, row['binary_label']

    def get_class_weights(self):
        counts = self.df['binary_label'].value_counts().sort_index().values
        total = len(self.df)
        return torch.FloatTensor(total / (len(counts) * counts))

# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ============================================================
# RESNET-18 WITH MC DROPOUT
# ============================================================
class ResNetWithUncertainty(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features  # = 512

        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def mc_predict(self, x, n_samples=20):
        self.eval()
        self.enable_dropout()

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                probs = torch.softmax(self(x), dim=1)
                preds.append(probs)

        preds = torch.stack(preds)
        mean_pred = preds.mean(dim=0)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)
        variance = preds.var(dim=0).mean(dim=1)

        return mean_pred, entropy, variance

# ============================================================
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ============================================================
# UTILS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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
    class_correct = torch.zeros(2)
    class_total = torch.zeros(2)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

            for c in range(2):
                mask = y == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == y[mask]).sum().item()
                    class_total[c] += mask.sum().item()

    per_class = (class_correct / (class_total + 1e-10)) * 100
    return total_loss / total, 100 * correct / total, per_class

# ============================================================
# TRAIN SINGLE MODEL
# ============================================================
def train_single_model(seed, model_idx, train_loader, val_loader, class_weights):
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL {model_idx+1}/5 — SEED {seed}")
    print(f"{'='*70}")

    set_seed(seed)
    model = ResNetWithUncertainty(NUM_CLASSES, DROPOUT_RATE).to(DEVICE)

    criterion = FocalLoss(alpha=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_acc = 0
    patience = 0

    for epoch in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train: Loss={tr_loss:.4f}, Acc={tr_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Per-class: Poor={per_class[0]:.1f}%, Good={per_class[1]:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/{TARGET_SCORE}_seed{seed}_best.pth"
            )
            print("   🎯 New best — model saved")
        else:
            patience += 1
            if patience >= PATIENCE:
                print("   🛑 Early stopping")
                break
        print()

    return best_acc

# ============================================================
# MAIN
# ============================================================
def main():
    dataset = GardnerDataset(TRAIN_CSV, IMG_FOLDER, TARGET_SCORE)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_weights = dataset.get_class_weights().to(DEVICE)

    results = []
    for i, seed in enumerate(SEEDS):
        acc = train_single_model(seed, i, train_loader, val_loader, class_weights)
        results.append(acc)

    print("\nAverage ensemble validation accuracy:", np.mean(results))

if __name__ == "__main__":
    main()
