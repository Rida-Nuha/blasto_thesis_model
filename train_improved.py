"""
Improved Training for 80%+ Accuracy
Key changes:
1. Unfreeze more layers
2. Higher learning rate
3. Better data augmentation
4. Longer training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
import pandas as pd
import os
import random
import numpy as np

from dataset import BlastocystDataset, train_transform, val_transform


# ============================================================
# IMPROVED MODEL
# ============================================================
class SwinUncertaintyImproved(nn.Module):
    """Improved model with partial unfreezing for better accuracy"""
    
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        print("Loading pretrained Swin-T...")
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        
        # Better head with batch norm
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        # CRITICAL: Only freeze early layers (0, 1)
        # Train layers 2, 3 and head
        for i, layer in enumerate(self.backbone.features):
            if i < 2:  # Freeze only first 2 layers
                for param in layer.parameters():
                    param.requires_grad = False
        
        print("✓ Frozen layers: 0, 1 (training layers 2, 3 + head)")
        self._print_params()
    
    def _print_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict_with_uncertainty(self, x, n_samples=10):
        self.train()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        self.eval()
        return {'probs': mean_probs, 'uncertainty': entropy}


# ============================================================
# IMPROVED CONFIG
# ============================================================
TRAIN_CSV = "data/Gardner_train_silver.csv"
IMG_FOLDER = "data/images"
SAVE_PATH = "saved_models/swin_improved_best.pth"
METRICS_FILE = "training_metrics_improved.csv"

NUM_CLASSES = 3
DROPOUT_RATE = 0.3  # Reduced dropout
BATCH_SIZE = 32
EPOCHS = 100  # More epochs
LR = 1e-4  # Higher LR (was 5e-5)
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.8
NUM_WORKERS = 4
PATIENCE = 20  # More patience
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# UTILS
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    def __init__(self, patience=20, save_path='best_model.pth'):
        self.patience = patience
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss >= self.best_loss:
            self.counter += 1
            print(f"   EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
    
    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)
        print(f"   ✓ Saved (val_loss: {self.best_loss:.4f})")


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        if hasattr(self.dataset, 'indices'):
            real_idx = self.dataset.indices[idx]
            img, label = self.dataset.dataset[real_idx]
        else:
            img, label = self.dataset[idx]
        
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __len__(self):
        return len(self.dataset.indices) if hasattr(self.dataset, 'indices') else len(self.dataset)


# ============================================================
# TRAINING WITH FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


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
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / total, 100 * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
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
            pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / total, 100 * correct / total


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("IMPROVED TRAINING FOR 80%+ ACCURACY")
    print("="*70 + "\n")
    
    set_seed(SEED)
    print(f"Device: {DEVICE}\n")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = BlastocystDataset(TRAIN_CSV, IMG_FOLDER, transform=None)
    
    # Split
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_ds = TransformDataset(train_ds, train_transform)
    val_ds = TransformDataset(val_ds, val_transform)
    
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=False)
    
    # IMPROVED MODEL
    print("\nInitializing improved model...")
    model = SwinUncertaintyImproved(NUM_CLASSES, DROPOUT_RATE)
    model.to(DEVICE)
    
    # Focal Loss for class imbalance
    class_weights = full_dataset.get_class_weights().to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    print(f"\nUsing Focal Loss with weights: {class_weights.cpu().numpy()}")
    
    # Optimizer with layer-wise LR
    optimizer = optim.AdamW([
        {'params': model.backbone.features.parameters(), 'lr': LR * 0.1},  # Lower LR for backbone
        {'params': model.backbone.head.parameters(), 'lr': LR}  # Higher LR for head
    ], weight_decay=WEIGHT_DECAY)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    early_stopping = EarlyStopping(patience=PATIENCE, save_path=SAVE_PATH)
    
    metrics = []
    best_val_acc = 0
    
    print("\n" + "="*70)
    print("TRAINING START")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        lr = optimizer.param_groups[0]['lr']
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"LR: {lr:.2e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   🎯 New best accuracy: {best_val_acc:.2f}%")
        
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr
        })
        
        scheduler.step()
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("\n🛑 Early stopping")
            break
        print()
    
    # Save metrics
    pd.DataFrame(metrics).to_csv(METRICS_FILE, index=False)
    print(f"\n✅ Metrics saved: {METRICS_FILE}")
    
    # Load best
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Best Val Loss: {val_loss:.4f}")
    print(f"   Best Val Acc:  {val_acc:.2f}%")
    
    if val_acc >= 80:
        print(f"   ✅ EXCELLENT! Thesis-quality result!")
    elif val_acc >= 75:
        print(f"   ✓ Good result, close to thesis quality")
    else:
        print(f"   ⚠️ Need improvement - see suggestions below")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
