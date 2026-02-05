"""
ResNet-18 with MC Dropout for Uncertainty Estimation
KAGGLE READY – ARCH-SPECIFIC CHECKPOINTS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os, random

# ================= CONFIG =================
ARCH = "resnet18"
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

# ================= DATASET =================
class GardnerDataset(Dataset):
    def __init__(self, csv, img_dir, target, transform=None):
        self.df = pd.read_csv(csv, sep=';')
        self.df = self.df[self.df[target].notna() & ~self.df[target].isin(['ND','NA'])]
        self.df[target] = pd.to_numeric(self.df[target])
        self.df["label"] = (self.df[target] >= BINARY_THRESHOLD).astype(int)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row["Image"])).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, row["label"]

    def get_class_weights(self):
        counts = self.df["label"].value_counts().sort_index().values
        total = len(self.df)
        return torch.FloatTensor(total / (len(counts) * counts))

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= MODEL =================
class ResNet18MC(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(f),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(f, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x): return self.backbone(x)

# ================= LOSS =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.cross_entropy(x, y, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()

# ================= TRAIN =================
def train_one(seed, train_loader, val_loader, class_weights):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model = ResNet18MC().to(DEVICE)
    crit = FocalLoss(alpha=class_weights)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best, patience = 0, 0

    for ep in range(EPOCHS):
        model.train()
        for x,y in train_loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        val_acc = 0; total = 0
        model.eval()
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                val_acc += (model(x).argmax(1)==y).sum().item()
                total += y.size(0)

        val_acc = 100 * val_acc / total
        print(f"Epoch {ep+1}: Val Acc {val_acc:.2f}%")

        if val_acc > best:
            best = val_acc
            patience = 0
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/{ARCH}_{TARGET_SCORE}_seed{seed}_best.pth"
            )
        else:
            patience += 1
            if patience >= PATIENCE: break

# ================= MAIN =================
def main():
    ds = GardnerDataset(TRAIN_CSV, IMG_FOLDER, TARGET_SCORE)
    tr_len = int(TRAIN_SPLIT*len(ds))
    tr, va = random_split(ds, [tr_len, len(ds)-tr_len])
    tr.dataset.transform = train_tf
    va.dataset.transform = val_tf

    tr_loader = DataLoader(tr, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_loader = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    weights = ds.get_class_weights().to(DEVICE)
    for s in SEEDS:
        train_one(s, tr_loader, va_loader, weights)

if __name__ == "__main__":
    main()
