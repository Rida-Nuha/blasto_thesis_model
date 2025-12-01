"""
Training Script: Swin Transformer with Uncertainty
Main training loop for thesis model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from tqdm import tqdm
import pandas as pd
import os
import sys
import random
import numpy as np

# Add src to path
sys.path.append('src')

from dataset import BlastocystDataset, train_transform, val_transform, create_dataloaders
from models.swin_uncertainty import SwinUncertainty


# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Paths
    TRAIN_CSV = "data/Gardner_train_silver.csv"
    IMG_FOLDER = "data/images"
    SAVE_DIR = "saved_models/uncertainty"
    METRICS_FILE = "results/metrics/uncertainty_training.csv"
    
    # Model
    NUM_CLASSES = 3
    DROPOUT_RATE = 0.5
    FREEZE_BACKBONE = True
    
    # Training
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 5e-5
    WEIGHT_DECAY = 1e-3
    
    # Data
    TRAIN_SPLIT = 0.8
    NUM_WORKERS = 4
    
    # Reproducibility
    SEED = 42
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Random seed set to {seed}")


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001, save_path='best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss - self.min_delta:
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
        print(f"   ✓ Model saved (val_loss: {self.best_loss:.4f})")


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Metrics
        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================
def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward
            logits = model(images)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def main():
    print("\n" + "="*70)
    print("SWIN TRANSFORMER WITH UNCERTAINTY - TRAINING")
    print("="*70 + "\n")
    
    # Set seed
    set_seed(Config.SEED)
    
    # Device info
    print(f"Device: {Config.DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("Running on CPU\n")
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = BlastocystDataset(
        csv_path=Config.TRAIN_CSV,
        root_dir=Config.IMG_FOLDER,
        transform=None  # Will apply in dataloader
    )
    
    # Split train/val
    train_size = int(Config.TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Apply transforms
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
    
    train_dataset = TransformDataset(train_dataset, train_transform)
    val_dataset = TransformDataset(val_dataset, val_transform)
    
    # Create dataloaders
    print(f"\nCreating dataloaders...")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=False
    )
    
    # Create model
    print("\nInitializing model...")
    model = SwinUncertainty(
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
        freeze_backbone=Config.FREEZE_BACKBONE
    )
    model.to(Config.DEVICE)
    
    # Loss function with class weights
    class_weights = full_dataset.get_class_weights().to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"\nClass weights: {class_weights.cpu().numpy()}")
    
    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE,
        min_delta=Config.MIN_DELTA,
        save_path=os.path.join(Config.SAVE_DIR, 'best_model.pth')
    )
    
    # Metrics storage
    metrics = []
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, Config.DEVICE
        )
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"LR: {current_lr:.2e}")
        
        # Save metrics
        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\n🛑 Early stopping triggered")
            break
        
        print()
    
    # Save final metrics
    os.makedirs(os.path.dirname(Config.METRICS_FILE), exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(Config.METRICS_FILE, index=False)
    print(f"\n✅ Metrics saved to: {Config.METRICS_FILE}")
    
    # Load best model
    model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, 'best_model.pth')))
    
    # Final validation
    val_loss, val_acc = validate(model, val_loader, criterion, Config.DEVICE)
    print(f"\n📊 FINAL RESULTS:")
    print(f"   Best Val Loss: {val_loss:.4f}")
    print(f"   Best Val Acc:  {val_acc:.2f}%")
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()