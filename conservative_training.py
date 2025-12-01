import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision.models import swin_t, Swin_T_Weights
from tqdm import tqdm
import pandas as pd
import os
import random
import numpy as np

from dataset import BlastocystDataset, train_transform, val_transform


# ============================================================
# CONSERVATIVE CONFIG - BASELINE WITHOUT DISAGREEMENT
# AMD GPU OPTIMIZED
# ============================================================
# AMD GPU detection
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"🎮 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CUDA not available, using CPU")

NUM_CLASSES = 3
BATCH_SIZE = 4  # Reduced for AMD GPU memory constraints
EPOCHS = 50
LR = 1e-5  # Very low learning rate
WEIGHT_DECAY = 1e-3  # Strong L2 regularization
DROPOUT_RATE = 0.5  # Maximum dropout
LABEL_SMOOTHING = 0.2  # Strong label smoothing
METRICS_FILE = "training_metrics_baseline.csv"
SEED = 42

# AMD-specific optimizations
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False  # More stable for AMD
torch.backends.cudnn.deterministic = True


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0.001, path='saved_models/best_baseline.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"✅ Val loss improved ({self.val_loss_min:.4f} → {val_loss:.4f}). Saving...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class SimpleSwinModel(nn.Module):
    """Very conservative Swin model with heavy regularization."""
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        
        # Simple head with strong dropout
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


def soft_cross_entropy_with_smoothing(pred, soft_targets, smoothing=0.0):
    """Soft CE with optional label smoothing."""
    n_classes = pred.size(1)
    log_probs = torch.log_softmax(pred, dim=1)
    
    if smoothing > 0:
        # Apply label smoothing to soft targets
        soft_targets = soft_targets * (1 - smoothing) + smoothing / n_classes
    
    loss = -(soft_targets * log_probs).sum(dim=1)
    return loss


class TransformWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        if hasattr(self.dataset, 'indices') and hasattr(self.dataset, 'dataset'):
            real_idx = self.dataset.indices[idx]
            img, soft_labels, weights = self.dataset.dataset[real_idx]
        else:
            img, soft_labels, weights = self.dataset[idx]

        if self.transform:
            img = self.transform(img)
        return img, soft_labels, weights

    def __len__(self):
        if hasattr(self.dataset, 'indices'):
            return len(self.dataset.indices)
        else:
            return len(self.dataset)


def train_model():
    set_seed(SEED)

    SILVER_PATH = "data/Gardner_train_silver.csv"
    GOLD_PATH = "data/Gardner_test_gold.xlsx"
    IMG_FOLDER = "data/images"

    # Load dataset
    dataset = BlastocystDataset(
        SILVER_PATH, 
        GOLD_PATH, 
        IMG_FOLDER,
        label_smoothing=0.0,  # We'll apply it in loss instead
        transform=None
    )

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Weighted sampler
    class_counts = torch.zeros(NUM_CLASSES)
    for idx in train_ds.indices:
        _, soft_label, _ = dataset[idx]
        hard = torch.argmax(soft_label)
        class_counts[hard] += 1

    class_weights = 1. / (class_counts + 1e-6)
    train_sample_weights = [class_weights[torch.argmax(dataset[idx][1])] for idx in train_ds.indices]
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    # Dataloaders with AMD-optimized settings
    train_loader = DataLoader(
        TransformWrapper(train_ds, train_transform),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,  # AMD GPUs work better with 0 workers
        pin_memory=False,  # Disable for AMD
        persistent_workers=False
    )
    val_loader = DataLoader(
        TransformWrapper(val_ds, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # AMD GPUs work better with 0 workers
        pin_memory=False  # Disable for AMD
    )

    print(f"\n{'='*60}")
    print(f"BASELINE TRAINING (No Disagreement Learning)")
    print(f"{'='*60}")
    print(f"Dataset: {len(dataset)} samples")
    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Class distribution: {class_counts.tolist()}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Dropout: {DROPOUT_RATE}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"{'='*60}\n")

    # Simple model
    model = SimpleSwinModel(num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE)
    
    # Freeze all but last layer + head
    print("🔒 Freezing backbone (only training head)...")
    for name, param in model.backbone.named_parameters():
        if 'head' not in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)\n")
    
    model.to(DEVICE)
    
    # AMD GPU: Enable TF32 for better performance if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32 enabled for faster training")
    
    # Clear cache before training (important for AMD)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU cache cleared\n")

    # Optimizer with strong regularization
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # Conservative scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-7
    )
    
    early_stopping = EarlyStopping(patience=15, verbose=True)
    metrics = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for images, soft_labels, weights in train_loop:
            images = images.to(DEVICE, non_blocking=False)  # AMD: blocking transfers
            soft_labels = soft_labels.to(DEVICE, non_blocking=False)
            weights = weights.to(DEVICE, non_blocking=False)
            
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            outputs = model(images)
            
            # Simple loss with label smoothing
            loss_per_sample = soft_cross_entropy_with_smoothing(
                outputs, soft_labels, smoothing=LABEL_SMOOTHING
            )
            loss = (loss_per_sample * weights / weights.sum()).sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            total_train_loss += loss.item() * images.size(0)
            train_loop.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        total_val_loss = 0
        correct, total = 0, 0
        
        with torch.no_grad():
            for images, soft_labels, _ in val_loader:
                images = images.to(DEVICE)
                soft_labels = soft_labels.to(DEVICE)
                
                outputs = model(images)
                val_loss = soft_cross_entropy_with_smoothing(
                    outputs, soft_labels, smoothing=LABEL_SMOOTHING
                ).mean()
                
                total_val_loss += val_loss.item() * images.size(0)
                
                preds = torch.argmax(outputs, 1)
                gt = torch.argmax(soft_labels, 1)
                correct += (preds == gt).sum().item()
                total += gt.size(0)

        train_avg_loss = total_train_loss / train_size
        val_avg_loss = total_val_loss / val_size
        val_acc = 100 * correct / total

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_avg_loss,
            'val_loss': val_avg_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}")
        print(f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

        scheduler.step(val_avg_loss)
        early_stopping(val_avg_loss, model)
        
        if early_stopping.early_stop:
            print("\n🛑 Early stopping triggered.")
            break

    # Save results
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"\n📊 Metrics saved to {METRICS_FILE}")
    print(f"✅ Best validation loss: {early_stopping.val_loss_min:.4f}")


if __name__ == "__main__":
    train_model()