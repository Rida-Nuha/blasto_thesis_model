"""
Full Fine-tuning for 80%+ Accuracy - BALANCED CLASS WEIGHTS

Key changes:
1. Unfreeze ENTIRE backbone (100% trainable)
2. Two-stage training: freeze → unfreeze
3. Aggressive augmentation + Mixup
4. FIXED: Balanced sqrt weights (not too aggressive)
5. FIXED: Higher Stage 2 LR
6. FIXED: Swin-Small model (more capacity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import swin_s, Swin_S_Weights  # Changed to Swin-Small
from tqdm import tqdm
import pandas as pd
import os
import random
import numpy as np

from dataset import BlastocystDataset, train_transform, val_transform


# ============================================================
# MODEL
# ============================================================
class SwinFullFinetune(nn.Module):
    """Fully fine-tuned Swin Transformer - Small version"""
    
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        print("Loading Swin-S pretrained model...")
        self.backbone = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)  # Changed to Small
        in_features = self.backbone.head.in_features
        
        # Enhanced head
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all except head"""
        for name, param in self.backbone.named_parameters():
            if 'head' not in name:
                param.requires_grad = False
        print("✓ Backbone frozen")
    
    def unfreeze_backbone(self):
        """Unfreeze everything"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✓ Backbone unfrozen - full fine-tuning")
    
    def print_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")


# ============================================================
# CONFIG - OPTIMIZED FOR 80%+
# ============================================================
TRAIN_CSV = "data/Gardner_train_silver.csv"
IMG_FOLDER = "data/images"
SAVE_PATH = "saved_models/swin_full_finetune_best.pth"
METRICS_FILE = "training_metrics_full_finetune.csv"

NUM_CLASSES = 3
DROPOUT_RATE = 0.3
BATCH_SIZE = 24  # Optimized for Swin-S
STAGE1_EPOCHS = 25
STAGE2_EPOCHS = 30
STAGE1_LR = 2e-3
STAGE2_LR = 1e-4  # Increased from 5e-5
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.85
NUM_WORKERS = 2
PATIENCE = 12
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
    def __init__(self, patience=12, save_path='best_model.pth'):
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
# MIXUP AUGMENTATION
# ============================================================
def mixup_data(x, y, alpha=0.4):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
# LABEL SMOOTHING CROSS ENTROPY
# ============================================================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_probs = F.log_softmax(pred, dim=1)
        
        # One-hot encoding with smoothing
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)
        
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


# ============================================================
# TRAINING
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device, use_mixup=False):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Apply mixup 50% of the time in Stage 2
        if use_mixup and np.random.random() > 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (lam * (preds == labels_a).float() + (1 - lam) * (preds == labels_b).float()).sum().item()
            total += labels.size(0)
        else:
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
    
    # Per-class accuracy
    class_correct = torch.zeros(3)
    class_total = torch.zeros(3)
    
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
            
            # Per-class
            for c in range(3):
                mask = labels == c
                if mask.sum() > 0:
                    class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    class_total[c] += mask.sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
    
    per_class_acc = (class_correct / (class_total + 1e-10)) * 100
    
    return total_loss / total, 100 * correct / total, per_class_acc


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("TWO-STAGE FULL FINE-TUNING FOR 80%+ ACCURACY")
    print("="*70 + "\n")
    
    set_seed(SEED)
    print(f"Device: {DEVICE}\n")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = BlastocystDataset(TRAIN_CSV, IMG_FOLDER, transform=None)
    
    # Split with more training data
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_ds = TransformDataset(train_ds, train_transform)
    val_ds = TransformDataset(val_ds, val_transform)
    
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}")
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                             num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model
    print("\nInitializing model...")
    model = SwinFullFinetune(NUM_CLASSES, DROPOUT_RATE)
    model.to(DEVICE)
    
    # Loss - BALANCED SQRT CLASS WEIGHTING
    print("\n" + "="*70)
    print("COMPUTING BALANCED CLASS WEIGHTS")
    print("="*70)
    
    # Get class distribution from dataset
    counts = np.array([1305, 332, 391])  # Poor, Medium, Good
    print(f"Class counts: {counts}")
    print(f"  Class 0 (Poor): {counts[0]}")
    print(f"  Class 1 (Medium): {counts[1]}")
    print(f"  Class 2 (Good): {counts[2]}")
    
    # Square root balancing (gentler than inverse, stronger than none)
    sqrt_weights = np.sqrt(counts.sum() / counts)
    class_weights = torch.FloatTensor(sqrt_weights).to(DEVICE)
    
    print(f"\nBalanced sqrt weights: {class_weights.cpu().numpy()}")
    print("(These weights help minorities without killing any class)\n")
    
    metrics = []
    best_val_acc = 0
    
    # ========================================================================
    # STAGE 1: Train head only (frozen backbone)
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 1: TRAINING HEAD ONLY (Backbone Frozen)")
    print("="*70 + "\n")
    
    model.freeze_backbone()
    model.print_params()
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1, weight=class_weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STAGE1_EPOCHS)
    
    for epoch in range(STAGE1_EPOCHS):
        print(f"Stage 1 - Epoch {epoch+1}/{STAGE1_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, use_mixup=False)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"Per-class: {per_class.numpy().round(1)}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   🎯 New best: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), SAVE_PATH.replace('.pth', '_stage1_best.pth'))
        
        metrics.append({
            'stage': 1,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        scheduler.step()
        print()
    
    # ========================================================================
    # STAGE 2: Fine-tune everything
    # ========================================================================
    print("\n" + "="*70)
    print("STAGE 2: FULL FINE-TUNING (Backbone Unfrozen)")
    print("="*70 + "\n")
    
    model.unfreeze_backbone()
    model.print_params()
    
    # Use same balanced weights for Stage 2 with Focal Loss
    print(f"Stage 2 weights (same balanced): {class_weights.cpu().numpy()}")
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Layer-wise LR with better ratios
    optimizer = optim.AdamW([
        {'params': model.backbone.features[0:2].parameters(), 'lr': STAGE2_LR * 0.1},
        {'params': model.backbone.features[2:4].parameters(), 'lr': STAGE2_LR * 0.3},
        {'params': model.backbone.features[4:].parameters(), 'lr': STAGE2_LR * 0.5},
        {'params': model.backbone.norm.parameters(), 'lr': STAGE2_LR * 2},
        {'params': model.backbone.head.parameters(), 'lr': STAGE2_LR * 5}
    ], weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[STAGE2_LR * 0.1, STAGE2_LR * 0.3, STAGE2_LR * 0.5, STAGE2_LR * 2, STAGE2_LR * 5],
        epochs=STAGE2_EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    early_stopping = EarlyStopping(patience=PATIENCE, save_path=SAVE_PATH)
    
    for epoch in range(STAGE2_EPOCHS):
        print(f"Stage 2 - Epoch {epoch+1}/{STAGE2_EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE, use_mixup=True)
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
        print(f"Per-class: {per_class.numpy().round(1)}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"   🎯 NEW BEST: {best_val_acc:.2f}%")
            torch.save(model.state_dict(), SAVE_PATH.replace('.pth', '_best_acc.pth'))
        
        metrics.append({
            'stage': 2,
            'epoch': STAGE1_EPOCHS + epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
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
    
    # Load best model and final evaluation
    try:
        model.load_state_dict(torch.load(SAVE_PATH.replace('.pth', '_best_acc.pth'), weights_only=True))
        val_loss, val_acc, per_class = validate(model, val_loader, criterion, DEVICE)
    except:
        print("Using final model state")
    
    print(f"\n" + "="*70)
    print("📊 FINAL RESULTS:")
    print("="*70)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"\nPer-class Accuracy:")
    print(f"  Class 0 (Poor):   {per_class[0]:.1f}%")
    print(f"  Class 1 (Medium): {per_class[1]:.1f}%")
    print(f"  Class 2 (Good):   {per_class[2]:.1f}%")
    
    if best_val_acc >= 80:
        print(f"\n✅ EXCELLENT! Thesis-quality result!")
    elif best_val_acc >= 75:
        print(f"\n✓ Good! Close to thesis quality")
    else:
        print(f"\n⚠️ Consider ensemble methods or more data augmentation")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
