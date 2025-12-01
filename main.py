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
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4  # Reduced LR for fine-tuning
WEIGHT_DECAY = 1e-4  # Added L2 regularization
DROPOUT_RATE = 0.4  # Increased dropout (tune between 0.3-0.5)
DISAGREEMENT_WEIGHT = 0.3  # Weight for disagreement loss (tune between 0.1-0.5)
LABEL_SMOOTHING = 0.1  # Label smoothing factor (tune between 0.0-0.2)
METRICS_FILE = "training_metrics.csv"
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✨ Random seed set to {seed_value} for reproducibility.")


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, path='saved_models/best_swin_icm.pth'):
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
            print(f"Validation loss decreased ({self.val_loss_min:.4f} → {val_loss:.4f}). Saving model…")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ============================================================
# DISAGREEMENT LEARNING LOSS
# ============================================================
class DisagreementLoss(nn.Module):
    """
    Implements disagreement learning for multi-annotator labels.
    Combines classification loss with disagreement regularization.
    """
    def __init__(self, disagreement_weight=0.3):
        super().__init__()
        self.disagreement_weight = disagreement_weight
    
    def soft_cross_entropy(self, pred, soft_targets):
        """Standard soft cross-entropy."""
        log_probs = torch.log_softmax(pred, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1)
        return loss
    
    def disagreement_regularization(self, pred, soft_targets):
        """
        Encourages model to capture annotator disagreement.
        Uses entropy of soft labels as disagreement measure.
        High entropy = high disagreement = model should be less confident.
        """
        # Calculate entropy of soft labels (disagreement level)
        epsilon = 1e-10
        label_entropy = -(soft_targets * torch.log(soft_targets + epsilon)).sum(dim=1)
        
        # Calculate model prediction entropy
        pred_probs = torch.softmax(pred, dim=1)
        pred_entropy = -(pred_probs * torch.log(pred_probs + epsilon)).sum(dim=1)
        
        # Encourage model entropy to match label disagreement
        disagreement_loss = torch.abs(pred_entropy - label_entropy)
        return disagreement_loss
    
    def forward(self, pred, soft_targets):
        """
        Combined loss: classification + disagreement regularization
        """
        ce_loss = self.soft_cross_entropy(pred, soft_targets)
        dis_loss = self.disagreement_regularization(pred, soft_targets)
        
        total_loss = ce_loss + self.disagreement_weight * dis_loss
        return total_loss, ce_loss, dis_loss


# ============================================================
# IMPROVED SWIN TRANSFORMER WITH LABEL SMOOTHING
# ============================================================
class SwinTransformerWithSmoothing(nn.Module):
    """
    Swin Transformer with dropout and optional label smoothing
    """
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        
        # Enhanced head with more regularization
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================
# SAFE TRANSFORM WRAPPER
# ============================================================
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


# ============================================================
# MIXUP AUGMENTATION (Optional but recommended)
# ============================================================
def mixup_data(x, y, alpha=0.2):
    """Applies mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model():
    set_seed(SEED)

    SILVER_PATH = "data/Gardner_train_silver.csv"
    GOLD_PATH = "data/Gardner_test_gold.xlsx"
    IMG_FOLDER = "data/images"

    # Load dataset with label smoothing
    dataset = BlastocystDataset(
        SILVER_PATH, 
        GOLD_PATH, 
        IMG_FOLDER,
        label_smoothing=LABEL_SMOOTHING,
        transform=None  # Transform applied by wrapper
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Weighted sampler
    class_counts = torch.zeros(NUM_CLASSES)
    for idx in train_ds.indices:
        _, soft_label, _ = dataset[idx]
        hard = torch.argmax(soft_label)
        class_counts[hard] += 1

    class_weights = 1. / class_counts
    train_sample_weights = [class_weights[torch.argmax(dataset[idx][1])] for idx in train_ds.indices]
    sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=True)

    # Dataloaders
    train_loader = DataLoader(
        TransformWrapper(train_ds, train_transform),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        persistent_workers=False
    )
    val_loader = DataLoader(
        TransformWrapper(val_ds, val_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Dataset loaded → {len(dataset)} samples")
    print(f"Train: {train_size}, Val: {val_size}")
    print(f"Class distribution: {class_counts.tolist()}")

    # Model setup with improved architecture
    print("\n🔧 Loading Swin Transformer with Disagreement Learning…")
    model = SwinTransformerWithSmoothing(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    )
    model.to(DEVICE)

    # Optimizer with weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR * 10,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    
    # Disagreement loss
    criterion = DisagreementLoss(disagreement_weight=DISAGREEMENT_WEIGHT)
    early_stopping = EarlyStopping(patience=10, verbose=True)

    metrics = []
    print(f"\n🔥 Training with Disagreement Learning (λ={DISAGREEMENT_WEIGHT})…\n")

    # ============================================================
    # EPOCH LOOP
    # ============================================================
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        total_ce_loss = 0
        total_dis_loss = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)

        for images, soft_labels, weights in train_loop:
            images = images.to(DEVICE)
            soft_labels = soft_labels.to(DEVICE)
            weights = weights.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Compute disagreement loss
            loss_per_sample, ce_per_sample, dis_per_sample = criterion(outputs, soft_labels)
            
            # Apply sample weights
            loss = (loss_per_sample * weights / weights.sum()).sum()
            ce_loss = (ce_per_sample * weights / weights.sum()).sum()
            dis_loss = (dis_per_sample * weights / weights.sum()).sum()
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item() * images.size(0)
            total_ce_loss += ce_loss.item() * images.size(0)
            total_dis_loss += dis_loss.item() * images.size(0)
            
            train_loop.set_postfix(
                loss=loss.item(),
                ce=ce_loss.item(),
                dis=dis_loss.item()
            )

        # ============================================================
        # VALIDATION
        # ============================================================
        model.eval()
        total_val_loss, total_val_ce, total_val_dis = 0, 0, 0
        correct, total = 0, 0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)

        with torch.no_grad():
            for images, soft_labels, _ in val_loop:
                images = images.to(DEVICE)
                soft_labels = soft_labels.to(DEVICE)
                
                outputs = model(images)
                val_loss, val_ce, val_dis = criterion(outputs, soft_labels)
                
                total_val_loss += val_loss.mean().item() * images.size(0)
                total_val_ce += val_ce.mean().item() * images.size(0)
                total_val_dis += val_dis.mean().item() * images.size(0)
                
                preds = torch.argmax(outputs, 1)
                gt = torch.argmax(soft_labels, 1)
                correct += (preds == gt).sum().item()
                total += gt.size(0)
                
                val_loop.set_postfix(v_loss=val_loss.mean().item())

        train_avg_loss = total_train_loss / train_size
        train_ce_loss = total_ce_loss / train_size
        train_dis_loss = total_dis_loss / train_size
        
        val_avg_loss = total_val_loss / val_size
        val_ce_loss = total_val_ce / val_size
        val_dis_loss = total_val_dis / val_size
        val_acc = 100 * correct / total

        metrics.append({
            'epoch': epoch + 1,
            'train_loss': train_avg_loss,
            'train_ce': train_ce_loss,
            'train_dis': train_dis_loss,
            'val_loss': val_avg_loss,
            'val_ce': val_ce_loss,
            'val_dis': val_dis_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_avg_loss:.4f} (CE: {train_ce_loss:.4f}, Dis: {train_dis_loss:.4f})")
        print(f"Val Loss: {val_avg_loss:.4f} (CE: {val_ce_loss:.4f}, Dis: {val_dis_loss:.4f})")
        print(f"Val Acc: {val_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 70)

        early_stopping(val_avg_loss, model)
        if early_stopping.early_stop:
            print("\n🚨 Early stopping triggered.")
            break

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(METRICS_FILE, index=False)
    print(f"📈 Metrics saved → {METRICS_FILE}")

    model.load_state_dict(torch.load(early_stopping.path))
    torch.save(model.state_dict(), "saved_models/swin_transformer_disagreement.pth")
    print(f"✅ Training complete. Best val loss: {early_stopping.val_loss_min:.4f}")


if __name__ == "__main__":
    train_model()