"""
DenseNet121 (RadImageNet Pre-trained)
Standard Categorical Architecture with Dynamic Class Weights
The "Brute Force" Medical Baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import densenet121
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
SAVE_DIR = "/kaggle/working/saved_models/densenet_baseline"        

# 🚨 PASTE YOUR RADIMAGENET DENSENET121 .PT PATH HERE
RADIMAGENET_WEIGHTS_PATH = "/kaggle/input/models/ridakhan09/embryo-grading-pretrained-weights/pytorch/base-weights/1/DenseNet121.pt"
NUM_CLASSES_EXP = 5  
NUM_CLASSES_ICM = 3  
NUM_CLASSES_TE = 3   

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
# 2. DATASET (Strict 0-Indexed Categorical Labels)
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
        
        # CrossEntropyLoss requires 0-indexed integers
        exp_val = int(row["EXP_silver"])
        l_exp = torch.tensor(exp_val - 1 if exp_val >= 1 else exp_val, dtype=torch.long)
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
# 3. DENSENET121 MODEL
# ============================================================
class MultiTaskRadImageNetDenseNet(nn.Module):
    def __init__(self, weight_path, dropout_rate=0.3):
        super().__init__()
        
        # Load empty DenseNet121
        self.backbone = densenet121(weights=None)
        
        # Inject RadImageNet Weights safely
        if os.path.exists(weight_path):
            print(f"Injecting RadImageNet Weights from: {weight_path}")
            state_dict = torch.load(weight_path, map_location='cpu', weights_only=False)
            
            # Clean module prefixes and drop the old classification head
            clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            clean_state_dict = {k: v for k, v in clean_state_dict.items() if not k.startswith('classifier.')}
            
            self.backbone.load_state_dict(clean_state_dict, strict=False)
        else:
            print(f"⚠️ WARNING: RadImageNet weights not found at {weight_path}. Training from scratch!")
        
        # DenseNet121 outputs 1024 features
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
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
# 4. TRAINING LOOP & DYNAMIC WEIGHT CALCULATOR
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(df, column_name, num_classes):
    """Calculates inverse frequency weights to penalize majority classes"""
    if column_name == "EXP_silver":
        counts = (df[column_name] - 1).value_counts().sort_index()
    else:
        counts = df[column_name].value_counts().sort_index()
        
    class_counts = np.zeros(num_classes)
    for k, v in counts.items():
        if 0 <= int(k) < num_classes:
            class_counts[int(k)] = v
            
    # Inverse frequency (add 1e-5 to avoid division by zero)
    weights = 1.0 / (class_counts + 1e-5)
    weights = weights / weights.sum()  # Normalize
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def train_epoch(model, loader, crit_exp, crit_icm, crit_te, optimizer, device):
    model.train()
    total_loss, correct_exp, correct_icm, correct_te, total = 0, 0, 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, l_exp, l_icm, l_te in pbar:
        images = images.to(device)
        l_exp, l_icm, l_te = l_exp.to(device), l_icm.to(device), l_te.to(device)
        
        optimizer.zero_grad()
        out_exp, out_icm, out_te = model(images)
        
        loss = crit_exp(out_exp, l_exp) + crit_icm(out_icm, l_icm) + crit_te(out_te, l_te)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        pred_exp = torch.argmax(out_exp, dim=1)
        pred_icm = torch.argmax(out_icm, dim=1)
        pred_te = torch.argmax(out_te, dim=1)
        
        correct_exp += (pred_exp == l_exp).sum().item()
        correct_icm += (pred_icm == l_icm).sum().item()
        correct_te += (pred_te == l_te).sum().item()
        total += l_exp.size(0)
        
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def validate(model, loader, crit_exp, crit_icm, crit_te, device):
    model.eval()
    total_loss, correct_exp, correct_icm, correct_te, total = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, l_exp, l_icm, l_te in pbar:
            images = images.to(device)
            l_exp, l_icm, l_te = l_exp.to(device), l_icm.to(device), l_te.to(device)
            
            out_exp, out_icm, out_te = model(images)
            loss = crit_exp(out_exp, l_exp) + crit_icm(out_icm, l_icm) + crit_te(out_te, l_te)
            total_loss += loss.item() * images.size(0)
            
            pred_exp = torch.argmax(out_exp, dim=1)
            pred_icm = torch.argmax(out_icm, dim=1)
            pred_te = torch.argmax(out_te, dim=1)
            
            correct_exp += (pred_exp == l_exp).sum().item()
            correct_icm += (pred_icm == l_icm).sum().item()
            correct_te += (pred_te == l_te).sum().item()
            total += l_exp.size(0)
            
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def train_single_model(seed, train_loader, val_loader, weights_dict):
    print(f"\n{'='*70}")
    print(f"TRAINING DENSENET121 (RADIMAGENET) WITH CLASS WEIGHTS (SEED {seed})")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    model = MultiTaskRadImageNetDenseNet(RADIMAGENET_WEIGHTS_PATH, DROPOUT_RATE).to(DEVICE)
    
    # Apply dynamic weights directly to CrossEntropy
    crit_exp = nn.CrossEntropyLoss(weight=weights_dict['exp'])
    crit_icm = nn.CrossEntropyLoss(weight=weights_dict['icm'])
    crit_te  = nn.CrossEntropyLoss(weight=weights_dict['te'])
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_avg_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        t_loss, (t_exp, t_icm, t_te) = train_epoch(model, train_loader, crit_exp, crit_icm, crit_te, optimizer, DEVICE)
        v_loss, (v_exp, v_icm, v_te) = validate(model, val_loader, crit_exp, crit_icm, crit_te, DEVICE)
        
        avg_val_acc = (v_exp + v_icm + v_te) / 3.0
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {t_loss:.4f} | Acc: EXP={t_exp:.1f}%, ICM={t_icm:.1f}%, TE={t_te:.1f}%")
        print(f"  Val Loss:   {v_loss:.4f} | Acc: EXP={v_exp:.1f}%, ICM={v_icm:.1f}%, TE={v_te:.1f}%")
        print(f"  --> Average Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"densenet_seed{seed}_best.pth"))
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
    
    # Safely extract labels strictly from the training split to calculate weights
    train_df = train_ds.dataset.df.iloc[train_ds.indices]
    weights_dict = {
        'exp': compute_class_weights(train_df, "EXP_silver", NUM_CLASSES_EXP),
        'icm': compute_class_weights(train_df, "ICM_silver", NUM_CLASSES_ICM),
        'te':  compute_class_weights(train_df, "TE_silver", NUM_CLASSES_TE)
    }
    
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    
    # Standard loaders (no sampler needed, loss function handles the imbalance)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    train_single_model(SEEDS[0], train_loader, val_loader, weights_dict)

if __name__ == "__main__":
    main()
