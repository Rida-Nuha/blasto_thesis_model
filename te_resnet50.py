import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import random
import os

# ============================================================
# CONFIGURATION - IDENTICAL TO SWIN VERSION
# ============================================================
SEEDS = [42, 123, 456, 789, 2024]
CSV_FILE = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
TARGET = "TE_silver"
SAVE_DIR = "saved_models/uncertainty_TE_resnet50"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ============================================================
# SEED FUNCTION
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
# MODEL - RESNET 50 (ONLY CHANGE)
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
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "="*70)
print("LOADING TE DATA")
print("="*70)

df = pd.read_csv(CSV_FILE, sep=";")
print(f"✓ Total samples: {len(df)}")

te_df = df[df[TARGET].notna()].copy()
te_df = te_df[~te_df[TARGET].isin(["ND", "NA"])]
te_df[TARGET] = pd.to_numeric(te_df[TARGET], errors="coerce")
te_df = te_df[te_df[TARGET].notna()]

print(f"✓ Valid TE samples: {len(te_df)}")
print(te_df[TARGET].value_counts().sort_index())

# Binary labels
te_df["label"] = te_df[TARGET].apply(lambda x: 1 if x >= 2 else 0)

# ============================================================
# TRANSFORMS (UNCHANGED)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
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
# TRAINING LOOP (UNCHANGED)
# ============================================================
all_results = []

for seed in SEEDS:
    print(f"\n{'='*70}")
    print(f"SEED {seed}")
    print(f"{'='*70}")

    set_seed(seed)

    train_df, val_df = train_test_split(
        te_df, test_size=0.2, stratify=te_df["label"], random_state=seed
    )

    class_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    class_weights = torch.tensor([
        total / (2.0 * class_counts[0]),
        total / (2.0 * class_counts[1])
    ]).to(device)

    train_loader = DataLoader(
        EmbryoDataset(train_df, IMG_FOLDER, train_transform),
        batch_size=32, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        EmbryoDataset(val_df, IMG_FOLDER, val_transform),
        batch_size=32, shuffle=False, num_workers=2
    )

    model = ResNetEmbryoClassifier(num_classes=2, dropout=0.4).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=25, eta_min=1e-6
    )

    best_acc = 0
    patience, patience_counter = 5, 0

    for epoch in range(25):
        model.train()
        train_preds, train_labels = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_preds.extend(out.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

        model.eval()
        val_preds, val_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.numpy())

        acc = accuracy_score(val_labels, val_preds)
        scheduler.step()

        print(f"Epoch {epoch+1} | Val Acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                f"{SAVE_DIR}/TE_resnet50_seed{seed}_best.pth"
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("⚠️ Early stopping")
            break

    all_results.append(best_acc)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("RESNET-50 TE ENSEMBLE RESULTS")
print("="*70)
print(f"Average Accuracy: {np.mean(all_results)*100:.2f}% ± {np.std(all_results)*100:.2f}%")
print("✅ TRAINING COMPLETE")
