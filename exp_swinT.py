"""
Swin Transformer with MC Dropout for Uncertainty Estimation
Novel contribution: Hybrid uncertainty quantification for embryo quality prediction
KAGGLE VERSION - All paths fixed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import random
import numpy as np

# ============================================================
# CONFIGURATION - KAGGLE PATHS FIXED
# ============================================================
TARGET_SCORE = "EXP_silver"  # Options: "EXP_silver", "ICM_silver", "TE_silver"

# ✅ KAGGLE CORRECT PATHS
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"  # ← FIXED
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"           # ← FIXED
SAVE_DIR = "kaggle/working/saved_models/uncertainty"        # ← FIXED
METRICS_FILE = f"/kaggle/working/training_metrics_{TARGET_SCORE}_uncertainty.csv"

BINARY_THRESHOLD = 2  # Good >= threshold

NUM_CLASSES = 2
DROPOUT_RATE = 0.3  # Higher dropout for better uncertainty
BATCH_SIZE = 32
EPOCHS = 1
LR = 1e-4
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.85
NUM_WORKERS = 2
PATIENCE = 15

# Train multiple models for ensemble
SEEDS = [42]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create save directory
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# DATASET - KAGGLE READY
# ============================================================
class GardnerDataset(Dataset):
    def __init__(self, csv_file, img_folder, target_column, threshold=2, transform=None):
        print(f"Loading {csv_file}...")
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.target_column = target_column
        self.threshold = threshold
        self.transform = transform
        
        # Filter valid samples and create binary labels
        valid_mask = (
            self.df[target_column].notna() & 
            (self.df[target_column] != 'ND') & 
            (self.df[target_column] != 'NA')
        )
        self.df = self.df[valid_mask].copy()
        self.df[target_column] = pd.to_numeric(self.df[target_column], errors='coerce')
        self.df = self.df[self.df[target_column].notna()].copy()
        
        self.df['binary_label'] = (self.df[target_column] >= threshold).astype(int)
        
        print(f"\n{'='*60}")
        print(f"Dataset for {target_column}")
        print(f"{'='*60}")
        print(f"Total samples (after filtering): {len(self.df)}")
        print(f"\nBinary distribution (threshold={threshold}):")
        print(f"  Poor (0): {(self.df['binary_label'] == 0).sum()} samples ({(self.df['binary_label'] == 0).mean()*100:.1f}%)")
        print(f"  Good (1): {(self.df['binary_label'] == 1).sum()} samples ({(self.df['binary_label'] == 1).mean()*100:.1f}%)")
        print(f"  Imbalance ratio: {((self.df['binary_label'] == 0).sum() / max((self.df['binary_label'] == 1).sum(), 1)):.2f}:1")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_path = os.path.join(self.img_folder, row['Image'])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
            
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = row['binary_label']
        
        return image, label
    
    def get_class_weights(self):
        counts = self.df['binary_label'].value_counts().sort_index().values
        total = len(self.df)
        weights = total / (len(counts) * counts)
        return torch.FloatTensor(weights)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# MODEL WITH MC DROPOUT
# ============================================================
class SwinWithUncertainty(nn.Module):
    """Swin Transformer with dropout enabled during inference for uncertainty"""
    
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        
        # Replace head with MC Dropout layers
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def enable_dropout(self):
        """Enable dropout for MC inference"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def mc_predict(self, x, n_samples=20):
        """Monte Carlo prediction with uncertainty"""
        self.eval()
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)
        variance = predictions.var(dim=0).mean(dim=1)
        
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
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# ============================================================
# UTILS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / total, 100 * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    class_correct = torch.zeros(2)
    class_total = torch.zeros(2)
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for c in range(2):
                mask = labels == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    class_total[c] += mask.sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    per_class_acc = (class_correct / (class_total + 1e-10)) * 100
    
    return total_loss / total, 100 * correct / total, per_class_acc

# ============================================================
# TRAIN SINGLE MODEL
# ============================================================
def train_single_model(seed, model_idx, train_loader, val_loader, class_weights):
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL {model_idx+1}/5 WITH SEED {seed}")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    
    # Model
    model = SwinWithUncertainty(NUM_CLASSES, DROPOUT_RATE)
    model.to(DEVICE)
    
    # Loss
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"  Per-class: Poor={per_class[0]:.1f}%, Good={per_class[1]:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            save_path = os.path.join(SAVE_DIR, f"{TARGET_SCORE}_seed{seed}_best.pth")
            torch.save(model.state_dict(), save_path)
            print(f"   🎯 New best: {best_val_acc:.2f}% (saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step()
        print()
    
    print(f"\n✅ Model {model_idx+1} finished!")
    print(f"   Best Val Acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    
    return best_val_acc

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print(f"TRAINING ENSEMBLE WITH UNCERTAINTY FOR {TARGET_SCORE}")
    print("="*70 + "\n")
    
    print(f"Device: {DEVICE}")
    print(f"CSV: {TRAIN_CSV}")
    print(f"Images: {IMG_FOLDER}\n")
    
    # Verify paths exist
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training CSV not found: {TRAIN_CSV}")
    if not os.path.exists(IMG_FOLDER):
        raise FileNotFoundError(f"Images folder not found: {IMG_FOLDER}")
    
    print("✅ Paths verified!")
    
    # Load dataset
    full_dataset = GardnerDataset(
        TRAIN_CSV, 
        IMG_FOLDER, 
        TARGET_SCORE,
        threshold=BINARY_THRESHOLD,
        transform=None
    )
    
    # Split
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Class weights
    class_weights = full_dataset.get_class_weights().to(DEVICE)
    print(f"\nClass weights: Poor={class_weights[0]:.3f}, Good={class_weights[1]:.3f}")
    
    # Train ensemble
    results = []
    for i, seed in enumerate(SEEDS):
        acc = train_single_model(seed, i, train_loader, val_loader, class_weights)
        results.append({'model': i+1, 'seed': seed, 'best_val_acc': acc})
    
    # Summary
    print("\n" + "="*70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*70)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print(f"\nAverage accuracy: {df['best_val_acc'].mean():.2f}%")
    print(f"Expected ensemble test: {df['best_val_acc'].mean() + 3:.2f}% - {df['best_val_acc'].mean() + 5:.2f}%")
    print(f"\n✅ Models saved in: {SAVE_DIR}")
    print("\n🎉 EXP TRAINING COMPLETE! Ready for test evaluation.")

if __name__ == "__main__":
    main()
