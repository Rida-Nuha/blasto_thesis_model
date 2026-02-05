"""
InceptionV3 with MC Dropout for Uncertainty Estimation
FULLY FIXED – Torchvision-safe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
import os, random

# ================= CONFIG =================
ARCH = "inceptionv3"
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
        df = pd.read_csv(csv, sep=";")
        df = df[df[target].notna() & ~df[target].isin(["ND", "NA"])]
        df[target] = pd.to_numeric(df[target])
        df["label"] = (df[target] >= BINARY_THRESHOLD).astype(int)

        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, r["Image"])).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, r["label"]

    def get_class_weights(self):
        c = self.df["label"].value_counts().sort_index().values
        return torch.FloatTensor(len(self.df) / (2 * c))

# ================= TRANSFORMS =================
train_tf = transforms.Compose([
    transforms.Resize((342,342)),
    transforms.RandomCrop(299),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ================= MODEL =================
class InceptionV3WithUncertainty(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True
        )

        self.backbone.AuxLogits = None
        f = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.BatchNorm1d(f),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(f, 512),
            nn.GELU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, NUM_CLASSES)
        )

    def forward(self, x):
        out = self.backbone(x)

        # 🔑 CRITICAL FIX
        if hasattr(out, "logits"):
            return out.logits

        return out

# ================= LOSS =================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

# ================= TRAIN =================
def train_epoch(model, loader, loss_fn, opt):
    model.train()
    loss_sum = correct = total = 0

    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()

        loss_sum += loss.item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)

    return loss_sum/total, 100*correct/total

def validate(model, loader, loss_fn):
    model.eval()
    loss_sum = correct = total = 0

    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss_sum/total, 100*correct/total

# ================= MAIN =================
def main():
    ds = GardnerDataset(TRAIN_CSV, IMG_FOLDER, TARGET_SCORE)
    tr_len = int(TRAIN_SPLIT * len(ds))
    tr, va = random_split(ds, [tr_len, len(ds)-tr_len])

    tr.dataset.transform = train_tf
    va.dataset.transform = val_tf

    tr_loader = DataLoader(tr, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    va_loader = DataLoader(va, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    weights = ds.get_class_weights().to(DEVICE)

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        print(f"\nTraining {ARCH} — seed {seed}")

        model = InceptionV3WithUncertainty().to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = FocalLoss(alpha=weights)

        best, patience = 0, 0
        for ep in range(EPOCHS):
            tr_loss, tr_acc = train_epoch(model, tr_loader, loss_fn, opt)
            va_loss, va_acc = validate(model, va_loader, loss_fn)

            print(f"Epoch {ep+1}: Train {tr_acc:.2f}% | Val {va_acc:.2f}%")

            if va_acc > best:
                best = va_acc
                patience = 0
                torch.save(
                    model.state_dict(),
                    f"{SAVE_DIR}/{ARCH}_{TARGET_SCORE}_seed{seed}_best.pth"
                )
            else:
                patience += 1
                if patience >= PATIENCE:
                    break

if __name__ == "__main__":
    main()
