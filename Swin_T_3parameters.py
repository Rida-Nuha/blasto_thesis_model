"""
Multi-Task Swin Transformer (Microscopy Pre-trained)
Featuring CORAL Ordinal Loss and MC Dropout Uncertainty
Final Master's Thesis Training Script
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
SAVE_DIR = "/kaggle/working/saved_models/swin_microscopy_coral"        

# 🚨 PASTE YOUR MICROSCOPY WEIGHTS KAGGLE PATH HERE
MICROSCOPY_WEIGHTS_PATH = "/kaggle/input/models/ridakhan09/embryo-grading-pretrained-weights/pytorch/base-weights/1/swin_tiny_patch4_window7_224_orig_Imge_micro.pth"

NUM_CLASSES_EXP = 5  
NUM_CLASSES_ICM = 3  # A, B, C (No NA)
NUM_CLASSES_TE = 3   # A, B, C (No NA)

DROPOUT_RATE = 0.3  
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-5  
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.85
NUM_WORKERS = 2
PATIENCE = 15

SEEDS = [42]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. CORAL LOSS FUNCTION & UTILS
# ============================================================
def to_coral_levels(label, num_classes):
    levels = [1] * label + [0] * (num_classes - 1 - label)
    return torch.tensor(levels, dtype=torch.float32)

class CoralLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, levels):
        val = (-torch.sum((F.logsigmoid(logits)*levels
                      + (F.logsigmoid(logits) - logits)*(1-levels)), dim=1))
        return torch.mean(val)

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
            
            # Keep only A (0), B (1), C (2)
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
        
        l_exp = to_coral_levels(int(row["EXP_silver"]), NUM_CLASSES_EXP)
        l_icm = to_coral_levels(int(row["ICM_silver"]), NUM_CLASSES_ICM)
        l_te = to_coral_levels(int(row["TE_silver"]), NUM_CLASSES_TE)
        
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
# 4. SWIN-T MODEL (MICROSCOPY WEIGHTS + CORAL HEADS)
# ============================================================
class MultiTaskMicroscopySwin(nn.Module):
    def __init__(self, weight_path, dropout_rate=0.3):
        super().__init__()
        self.backbone = swin_t(weights=None)
        
        if os.path.exists(weight_path):
            print(f"Injecting Microscopy Weights from: {weight_path}")
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
            
            # Extract the state dictionary
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            
            # 🚨 THE FIX: Delete the old 74-class head weights so PyTorch doesn't crash!
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head.')}
            
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            print(f"⚠️ WARNING: Microscopy weights not found. Training from scratch!")
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        self.exp_head = self._make_head(in_features, NUM_CLASSES_EXP - 1, dropout_rate)
        self.icm_head = self._make_head(in_features, NUM_CLASSES_ICM - 1, dropout_rate)
        self.te_head = self._make_head(in_features, NUM_CLASSES_TE - 1, dropout_rate)

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
        
        loss = criteria(out_exp, l_exp) + criteria(out_icm, l_icm) + criteria(out_te, l_te)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        pred_exp = (torch.sigmoid(out_exp) > 0.5).sum(dim=1)
        pred_icm = (torch.sigmoid(out_icm) > 0.5).sum(dim=1)
        pred_te = (torch.sigmoid(out_te) > 0.5).sum(dim=1)
        
        true_exp = l_exp.sum(dim=1)
        true_icm = l_icm.sum(dim=1)
        true_te = l_te.sum(dim=1)
        
        correct_exp += (pred_exp == true_exp).sum().item()
        correct_icm += (pred_icm == true_icm).sum().item()
        correct_te += (pred_te == true_te).sum().item()
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
            loss = criteria(out_exp, l_exp) + criteria(out_icm, l_icm) + criteria(out_te, l_te)
            total_loss += loss.item() * images.size(0)
            
            pred_exp = (torch.sigmoid(out_exp) > 0.5).sum(dim=1)
            pred_icm = (torch.sigmoid(out_icm) > 0.5).sum(dim=1)
            pred_te = (torch.sigmoid(out_te) > 0.5).sum(dim=1)
            
            true_exp = l_exp.sum(dim=1)
            true_icm = l_icm.sum(dim=1)
            true_te = l_te.sum(dim=1)
            
            correct_exp += (pred_exp == true_exp).sum().item()
            correct_icm += (pred_icm == true_icm).sum().item()
            correct_te += (pred_te == true_te).sum().item()
            total += l_exp.size(0)
            
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def train_single_model(seed, train_loader, val_loader):
    print(f"\n{'='*70}")
    print(f"TRAINING SWIN-T (MICROSCOPY) WITH CORAL LOSS (SEED {seed})")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    model = MultiTaskMicroscopySwin(MICROSCOPY_WEIGHTS_PATH, DROPOUT_RATE).to(DEVICE)
    criteria = CoralLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_avg_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        t_loss, (t_exp, t_icm, t_te) = train_epoch(model, train_loader, criteria, optimizer, DEVICE)
        v_loss, (v_exp, v_icm, v_te) = validate(model, val_loader, criteria, DEVICE)
        
        avg_val_acc = (v_exp + v_icm + v_te) / 3.0
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {t_loss:.4f} | Acc: EXP={t_exp:.1f}%, ICM={t_icm:.1f}%, TE={t_te:.1f}%")
        print(f"  Val Loss:   {v_loss:.4f} | Acc: EXP={v_exp:.1f}%, ICM={v_icm:.1f}%, TE={v_te:.1f}%")
        print(f"  --> Average Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"swin_coral_seed{seed}_best.pth"))
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    train_single_model(SEEDS[0], train_loader, val_loader)

if __name__ == "__main__":
    main()
