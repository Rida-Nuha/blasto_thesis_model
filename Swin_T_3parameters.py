"""
Multi-Task Swin Transformer (Microscopy Pre-trained)
Featuring "Full Focal" Architecture for Extreme Imbalance
Final Champion Master's Thesis Training Script
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import swin_t
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import random
import numpy as np

# ============================================================
# 1. CONFIGURATION
# ============================================================
TRAIN_CSV = "/kaggle/input/datasets/ridakhan09/dataset/Gardner_train_silver.csv"  
IMG_FOLDER = "/kaggle/input/datasets/ridakhan09/dataset/Images/Images"            
SAVE_DIR = "/kaggle/working/saved_models/swin_focal_champion"        

# 🚨 VERIFY YOUR MICROSCOPY WEIGHTS KAGGLE PATH HERE
MICROSCOPY_WEIGHTS_PATH = "/kaggle/input/models/ridakhan09/embryo-grading-pretrained-weights/pytorch/base-weights/1/swin_tiny_patch4_window7_224_orig_Imge_micro.pth"

NUM_CLASSES_EXP = 5  
NUM_CLASSES_ICM = 3  # A, B, C 
NUM_CLASSES_TE = 3   # A, B, C 

DROPOUT_RATE = 0.3  
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4  
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.85
NUM_WORKERS = 2
PATIENCE = 15

SEEDS = [42]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. FOCAL LOSS FUNCTION
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma # Gamma=2 strongly penalizes majority class guessing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ============================================================
# 3. DATASET
# ============================================================
class MultiTaskGardnerDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None):
        print(f"Loading {csv_file}...")
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.transform = transform
        
        self.targets = ["EXP_silver", "ICM_silver", "TE_silver"]
        
        for col in self.targets:
            valid_mask = (self.df[col].notna()) & (self.df[col] != 'ND') & (self.df[col] != 'NA')
            self.df = self.df[valid_mask].copy()
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(int)
            
            # Keep only the valid 3 classes for ICM/TE
            if col in ["ICM_silver", "TE_silver"]:
                self.df = self.df[self.df[col] < 3].copy() 
                
            self.df = self.df[self.df[col].notna()].copy()
            
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
        
        # 🚨 FULL FOCAL LABELS: All targets must be 0-indexed integers!
        exp_val = int(row["EXP_silver"])
        l_exp = torch.tensor(exp_val - 1 if exp_val >= 1 else exp_val, dtype=torch.long)
        
        # Assuming original dataset mapped A=0, B=1, C=2
        l_icm = torch.tensor(int(row["ICM_silver"]), dtype=torch.long)
        l_te = torch.tensor(int(row["TE_silver"]), dtype=torch.long)
        
        return image, l_exp, l_icm, l_te

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
# 4. SWIN-T MODEL (CATEGORICAL HEADS)
# ============================================================
class MultiTaskMicroscopySwin(nn.Module):
    def __init__(self, weight_path, dropout_rate=0.3):
        super().__init__()
        self.backbone = swin_t(weights=None)
        
        if os.path.exists(weight_path):
            print(f"Injecting Microscopy Weights from: {weight_path}")
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            # Strip the old heads to avoid dimension mismatch
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ WARNING: Microscopy weights not found. Training from scratch!")
            
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        # Standard categorical outputs for Focal Loss
        self.exp_head = self._make_head(in_features, NUM_CLASSES_EXP, dropout_rate)
        self.icm_head = self._make_head(in_features, NUM_CLASSES_ICM, dropout_rate)
        self.te_head = self._make_head(in_features, NUM_CLASSES_TE, dropout_rate)

    def _make_head(self, in_features, out_features, dropout_rate):
        return nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, out_features)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.exp_head(features), self.icm_head(features), self.te_head(features)

# ============================================================
# 5. TRAINING LOOP
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_epoch(model, loader, criteria, optimizer, device):
    model.train()
    total_loss, correct_exp, correct_icm, correct_te, total = 0, 0, 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, l_exp, l_icm, l_te in pbar:
        images = images.to(device)
        l_exp, l_icm, l_te = l_exp.to(device), l_icm.to(device), l_te.to(device)
        
        optimizer.zero_grad()
        out_exp, out_icm, out_te = model(images)
        
        # Apply Focal Loss to all heads
        loss_exp = criteria(out_exp, l_exp)
        loss_icm = criteria(out_icm, l_icm)
        loss_te  = criteria(out_te, l_te)
        
        loss = loss_exp + loss_icm + loss_te
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        # Standard Argmax Decoding
        pred_exp = torch.argmax(out_exp, dim=1)
        pred_icm = torch.argmax(out_icm, dim=1)
        pred_te = torch.argmax(out_te, dim=1)
        
        correct_exp += (pred_exp == l_exp).sum().item()
        correct_icm += (pred_icm == l_icm).sum().item()
        correct_te += (pred_te == l_te).sum().item()
        total += l_exp.size(0)
        
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def validate(model, loader, criteria, device):
    model.eval()
    total_loss, correct_exp, correct_icm, correct_te, total = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, l_exp, l_icm, l_te in pbar:
            images = images.to(device)
            l_exp, l_icm, l_te = l_exp.to(device), l_icm.to(device), l_te.to(device)
            
            out_exp, out_icm, out_te = model(images)
            
            loss_exp = criteria(out_exp, l_exp)
            loss_icm = criteria(out_icm, l_icm)
            loss_te  = criteria(out_te, l_te)
            
            loss = loss_exp + loss_icm + loss_te
            total_loss += loss.item() * images.size(0)
            
            pred_exp = torch.argmax(out_exp, dim=1)
            pred_icm = torch.argmax(out_icm, dim=1)
            pred_te = torch.argmax(out_te, dim=1)
            
            correct_exp += (pred_exp == l_exp).sum().item()
            correct_icm += (pred_icm == l_icm).sum().item()
            correct_te += (pred_te == l_te).sum().item()
            total += l_exp.size(0)
            
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def train_single_model(seed, train_loader, val_loader):
    print(f"\n{'='*70}")
    print(f"TRAINING SWIN-T WITH FULL FOCAL LOSS (SEED {seed})")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    model = MultiTaskMicroscopySwin(MICROSCOPY_WEIGHTS_PATH, DROPOUT_RATE).to(DEVICE)
    
    focal_criteria = FocalLoss(gamma=2.0) 
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_avg_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        t_loss, (t_exp, t_icm, t_te) = train_epoch(model, train_loader, focal_criteria, optimizer, DEVICE)
        v_loss, (v_exp, v_icm, v_te) = validate(model, val_loader, focal_criteria, DEVICE)
        
        avg_val_acc = (v_exp + v_icm + v_te) / 3.0
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {t_loss:.4f} | Acc: EXP={t_exp:.1f}%, ICM={t_icm:.1f}%, TE={t_te:.1f}%")
        print(f"  Val Loss:   {v_loss:.4f} | Acc: EXP={v_exp:.1f}%, ICM={v_icm:.1f}%, TE={v_te:.1f}%")
        print(f"  --> Average Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"swin_focal_seed{seed}_best.pth"))
            print(f"   🎯 New best average accuracy! (saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
                
        scheduler.step()
        print()
        
    print(f"\n✅ Training finished! Best Avg Acc: {best_avg_acc:.2f}% at epoch {best_epoch}")

def main():
    print(f"Device: {DEVICE}")
    full_dataset = MultiTaskGardnerDataset(TRAIN_CSV, IMG_FOLDER, transform=None)
    
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    
    # Standard DataLoaders - No samplers required!
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    train_single_model(SEEDS[0], train_loader, val_loader)

if __name__ == "__main__":
    main()
