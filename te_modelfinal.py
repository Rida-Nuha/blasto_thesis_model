"""
IMPROVED TE CLASSIFICATION MODEL
Target: Achieve >92% validation accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import random
import os

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
CSV_FILE = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
TARGET = 'TE_silver'
SAVE_DIR = "saved_models/improved_TE"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# Set seeds
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)

# ============================================================
# FOCAL LOSS (BETTER THAN CROSS-ENTROPY FOR HARD SAMPLES)
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================
# IMPROVED DATASET WITH STRONGER AUGMENTATION
# ============================================================
class EmbryoDataset(Dataset):
    def __init__(self, df, img_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Image']
        label = self.df.loc[idx, 'label']
        
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# ============================================================
# IMPROVED MODEL WITH HIGHER DROPOUT
# ============================================================
class ImprovedSwinTE(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):  # Increased from 0.4
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        # Deeper classifier with more regularization
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout * 0.8),  # Slightly lower for final layer
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================
print("\n" + "="*70)
print("LOADING TE DATA")
print("="*70)

df = pd.read_csv(CSV_FILE, sep=';')
print(f"✓ Total samples: {len(df)}")

# Filter valid TE values
te_df = df[df[TARGET].notna()].copy()
te_df = te_df[te_df[TARGET] != 'ND'].copy()
te_df = te_df[te_df[TARGET] != 'NA'].copy()
te_df[TARGET] = pd.to_numeric(te_df[TARGET], errors='coerce')
te_df = te_df[te_df[TARGET].notna()].copy()

print(f"✓ Valid TE samples: {len(te_df)}")
print(f"\nTE distribution:")
print(te_df[TARGET].value_counts().sort_index())

# Binary conversion (TE >= 2 is Good)
te_df['label'] = te_df[TARGET].apply(lambda x: 1 if x >= 2 else 0)

print(f"\nBinary label distribution:")
poor_count = (te_df['label'] == 0).sum()
good_count = (te_df['label'] == 1).sum()
print(f"  Poor (TE < 2): {poor_count} ({poor_count/len(te_df)*100:.1f}%)")
print(f"  Good (TE >= 2): {good_count} ({good_count/len(te_df)*100:.1f}%)")

# Calculate class weights (inverse frequency)
class_weights = torch.tensor([
    len(te_df) / (2 * poor_count),
    len(te_df) / (2 * good_count)
], dtype=torch.float32).to(device)

print(f"\nClass weights: Poor={class_weights[0]:.3f}, Good={class_weights[1]:.3f}")

# Train-validation split (80-20)
train_df, val_df = train_test_split(
    te_df, test_size=0.2, stratify=te_df['label'], random_state=SEED
)

print(f"\nTrain: {len(train_df)} | Validation: {len(val_df)}")

# ============================================================
# STRONGER DATA AUGMENTATION
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = EmbryoDataset(train_df, IMG_FOLDER, train_transform)
val_dataset = EmbryoDataset(val_df, IMG_FOLDER, val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# ============================================================
# INITIALIZE MODEL
# ============================================================
print("\n" + "="*70)
print("INITIALIZING IMPROVED TE MODEL")
print("="*70)

model = ImprovedSwinTE(num_classes=2, dropout=0.5).to(device)

# Focal Loss with class weights
criterion = FocalLoss(alpha=class_weights, gamma=2.0)

# Higher learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Cosine annealing with warm restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

print(f"✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"✓ Loss: Focal Loss (alpha={class_weights.tolist()}, gamma=2.0)")
print(f"✓ Optimizer: AdamW (lr=3e-4, weight_decay=0.01)")

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n" + "="*70)
print("TRAINING IMPROVED TE MODEL")
print("="*70)

NUM_EPOCHS = 30  # Increased from 25
best_val_acc = 0.0
patience = 8
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Training phase
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    train_acc = accuracy_score(train_labels, train_preds)
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_acc = accuracy_score(val_labels, val_preds)
    avg_val_loss = val_loss / len(val_loader)
    
    # Update learning rate
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% {'🎯 BEST!' if val_acc > best_val_acc else ''}")
    print(f"  LR: {current_lr:.2e}")
    
    # Confusion matrix every 5 epochs
    if (epoch + 1) % 5 == 0:
        cm = confusion_matrix(val_labels, val_preds)
        print(f"\n  Validation Confusion Matrix:")
        print(f"    Predicted: Poor  Good")
        print(f"    Poor:     {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"    Good:     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/TE_improved_seed{SEED}_best.pth")
        print(f"  ✅ Saved best model!")
        patience_counter = 0
    else:
        patience_counter += 1
    
    # Early stopping
    if patience_counter >= patience:
        print(f"\n⚠️ Early stopping triggered (patience={patience})")
        break

# ============================================================
# FINAL EVALUATION
# ============================================================
print("\n" + "="*70)
print("FINAL TE MODEL EVALUATION")
print("="*70)

# Load best model
model.load_state_dict(torch.load(f"{SAVE_DIR}/TE_improved_seed{SEED}_best.pth"))
model.eval()

val_preds, val_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        val_preds.extend(preds.cpu().numpy())
        val_labels.extend(labels.numpy())

val_acc = accuracy_score(val_labels, val_preds)
cm = confusion_matrix(val_labels, val_preds)

print(f"\n✅ Best Validation Accuracy: {val_acc*100:.2f}%")
print(f"\nFinal Confusion Matrix:")
print("             Predicted")
print("             Poor  Good")
print(f"Actual Poor {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Good {cm[1,0]:4d}  {cm[1,1]:4d}")

print("\nDetailed Classification Report:")
print(classification_report(val_labels, val_preds, 
                          target_names=['Poor (TE<2)', 'Good (TE>=2)'], 
                          digits=4))

print(f"\n🎯 Target: >92% | Achieved: {val_acc*100:.2f}%")
if val_acc >= 0.92:
    print("✅ TARGET ACHIEVED! Model ready for ensemble training.")
else:
    print(f"⚠️ Need {(0.92-val_acc)*100:.2f}% more improvement.")

print("\n" + "="*70)
print("✅ IMPROVED TE MODEL TRAINING COMPLETE!")
print("="*70)
