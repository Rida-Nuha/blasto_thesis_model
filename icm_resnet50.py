"""
ICM (Inner Cell Mass) Quality Prediction with Uncertainty Quantification
ResNet-50 + Deep Ensembles + MC Dropout
STRICTLY IDENTICAL TO SWIN VERSION EXCEPT BACKBONE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import os
import warnings
import shutil
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION (UNCHANGED)
# ============================================================
TARGET_SCORE = "ICM_silver"
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "saved_models/uncertainty_ICM_resnet50"
OUTPUT_DIR = "uncertainty_results_ICM"
BATCH_SIZE = 32
NUM_EPOCHS = 60
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
SEEDS = [42, 123, 456, 789, 2024]
PATIENCE = 20
MC_DROPOUT_SAMPLES = 50

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ============================================================
# DATASET (UNCHANGED)
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
# LOAD DATA (UNCHANGED)
# ============================================================
df = pd.read_csv(TRAIN_CSV, sep=";")
df["label"] = df[TARGET_SCORE].apply(lambda x: 1 if x >= 2 else 0)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df["label"]
)

class_counts = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (2 * class_counts)

if min(class_counts) / max(class_counts) < 0.25:
    class_weights[class_counts.argmin()] *= 1.5

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# ============================================================
# TRANSFORMS (UNCHANGED)
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
# MODEL — RESNET-50 (ONLY CHANGE)
# ============================================================
class ResNetEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()

        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # IDENTICAL HEAD TO SWIN VERSION
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
        return self.classifier(self.backbone(x))

# ============================================================
# FOCAL LOSS (UNCHANGED)
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
# TRAINING FUNCTION (UNCHANGED)
# ============================================================
def train_model(seed):
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

    model = ResNetEmbryoClassifier().to(device)
    criterion = FocalLoss(alpha=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc, patience_counter = 0, 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Seed {seed} Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds.extend(model(x).argmax(1).cpu().numpy())
                labels.extend(y.numpy())

        acc = accuracy_score(labels, preds)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                f"{MODEL_DIR}/ICM_resnet50_seed{seed}_best.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_acc

# ============================================================
# TRAIN ENSEMBLE
# ============================================================
results = {seed: train_model(seed) for seed in SEEDS}

# ============================================================
# SUMMARY
# ============================================================
print("\nRESNET-50 ICM RESULTS")
for s, a in results.items():
    print(f"Seed {s}: {a*100:.2f}%")

print(f"Average Accuracy: {np.mean(list(results.values()))*100:.2f}%")

# ============================================================
# ARCHIVE MODELS
# ============================================================
shutil.make_archive("ICM_resnet50_models", "zip", MODEL_DIR)
print("✅ Models archived: ICM_resnet50_models.zip")
