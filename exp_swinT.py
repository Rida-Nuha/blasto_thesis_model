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

NUM_CLASSES = 5  # Changed from 2 to 4 (labels 0, 1, 2, 3, 4)
DROPOUT_RATE = 0.3  # Higher dropout for better uncertainty
BATCH_SIZE = 32
EPOCHS = 50
LR = 2e-5
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
    def __init__(self, csv_file, img_folder, target_column,  transform=None):
        print(f"Loading {csv_file}...")
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.target_column = target_column
        self.transform = transform
        
        # Filter valid samples
        valid_mask = (
            self.df[target_column].notna() & 
            (self.df[target_column] != 'ND') & 
            (self.df[target_column] != 'NA')
        )
        self.df = self.df[valid_mask].copy()
        
        # Convert directly to integers
        self.df[target_column] = pd.to_numeric(self.df[target_column], errors='coerce').astype(int)
        self.df = self.df[self.df[target_column].notna()].copy()
        
        print(f"\n{'='*60}")
        print(f"Dataset for {target_column}")
        print(f"{'='*60}")
        print(f"Total samples (after filtering): {len(self.df)}")
        print(f"\nClass distribution:")
        print(self.df[target_column].value_counts().sort_index())
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
        
        # Fetch the original label
        label = row[self.target_column]
        
        return image, label
    
    def get_class_weights(self):
        # Safely ensure weights array always matches NUM_CLASSES exactly
        total = len(self.df)
        weights = np.ones(NUM_CLASSES, dtype=np.float32)
        counts = self.df[self.target_column].value_counts()
        
        for i in range(NUM_CLASSES):
            if i in counts:
                weights[i] = total / (NUM_CLASSES * counts[i])
            else:
                weights[i] = 1.0  # Safe fallback if a class is missing from the split
                
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
        images = images.to(device)
        # Force the dtype to long here!
        labels = labels.to(device, dtype=torch.long)
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
    
    # Change from 2 to NUM_CLASSES
    class_correct = torch.zeros(NUM_CLASSES)
    class_total = torch.zeros(NUM_CLASSES)
    
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
            
            # Change loop range to NUM_CLASSES
            for c in range(NUM_CLASSES):
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
   criterion = nn.CrossEntropyLoss(weight=class_weights)
    
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
      # Replace the hardcoded Poor/Good print statement with this:
        per_class_str = ", ".join([f"Class {i}={acc:.1f}%" for i, acc in enumerate(per_class)])
        print(f"  Per-class: {per_class_str}")
        
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
    
    # Print the weights for all 4 classes
    weights_str = ", ".join([f"Class {i}={w:.3f}" for i, w in enumerate(class_weights)])
    print(f"\nClass weights: {weights_str}")
    
    # Train ensemble
    
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
