"""
Multi-Task Swin Transformer with MC Dropout for Uncertainty Estimation
Journal-Worthy Architecture for Comprehensive Embryo Grading (EXP, ICM, TE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
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
# CONFIGURATION
# ============================================================
TRAIN_CSV = "/kaggle/input/datasets/ridakhan09/dataset/Gardner_train_silver.csv"  
IMG_FOLDER = "/kaggle/input/datasets/ridakhan09/dataset/Images/Images"            
SAVE_DIR = "/kaggle/working/saved_models/multitask"        

# Classes based on the clinical grading scale mappings
NUM_CLASSES_EXP = 5  # 0->1, 1->2, 2->3, 3->4, 4->5
NUM_CLASSES_ICM = 4  # 0->A, 1->B, 2->C, 3->ND
NUM_CLASSES_TE = 4   # 0->A, 1->B, 2->C, 3->ND

DROPOUT_RATE = 0.3  
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-5  # Stabilized learning rate for Swin fine-tuning
WEIGHT_DECAY = 5e-4
TRAIN_SPLIT = 0.85
NUM_WORKERS = 2
PATIENCE = 15

SEEDS = [42]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# DATASET
# ============================================================
class MultiTaskGardnerDataset(Dataset):
    def __init__(self, csv_file, img_folder, transform=None):
        print(f"Loading {csv_file}...")
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.transform = transform
        
        self.targets = ["EXP_silver", "ICM_silver", "TE_silver"]
        
        # Filter valid samples for ALL THREE targets simultaneously
        for col in self.targets:
            # We assume the CSV contains integers (0-4 for EXP, 0-3 for ICM/TE)
            valid_mask = (self.df[col].notna()) & (self.df[col] != 'ND') & (self.df[col] != 'NA')
            self.df = self.df[valid_mask].copy()
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(int)
            self.df = self.df[self.df[col].notna()].copy()
        
        print(f"\n{'='*60}")
        print("MULTI-TASK DATASET LOADED")
        print(f"Total valid multi-task samples: {len(self.df)}")
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
        
        # Fetch the original integer labels
        l_exp = row["EXP_silver"]
        l_icm = row["ICM_silver"]
        l_te = row["TE_silver"]
        
        return image, l_exp, l_icm, l_te
    
    def get_sample_weights(self):
        # We will base the sampling weights primarily on the ICM grade, 
        # as it is notoriously the most imbalanced in blastocyst datasets.
        icm_counts = self.df["ICM_silver"].value_counts().to_dict()
        
        # Calculate the weight for each individual image in the dataset
        sample_weights = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            icm_class = row["ICM_silver"]
            
            # The weight is the inverse of the class frequency
            # Rare classes get high weights, common classes get low weights
            weight = 1.0 / icm_counts.get(icm_class, 1.0)
            sample_weights.append(weight)
            
        return torch.DoubleTensor(sample_weights)

    def get_class_weights(self):
        total = len(self.df)
        
        def calc_weights(col_name, num_c):
            weights = np.ones(num_c, dtype=np.float32)
            counts = self.df[col_name].value_counts()
            for i in range(num_c):
                if i in counts:
                    weights[i] = total / (num_c * counts[i])
                else:
                    weights[i] = 1.0  
            return torch.FloatTensor(weights)
            
        w_exp = calc_weights("EXP_silver", NUM_CLASSES_EXP)
        w_icm = calc_weights("ICM_silver", NUM_CLASSES_ICM)
        w_te = calc_weights("TE_silver", NUM_CLASSES_TE)
        
        return w_exp, w_icm, w_te

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
# MULTI-TASK MODEL WITH LAYER NORM
# ============================================================
class MultiTaskSwinWithUncertainty(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        
        # Remove original single head
        self.backbone.head = nn.Identity()
        
        # Create Three Independent Heads using LayerNorm
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
    
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

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

def train_epoch(model, loader, criteria, optimizer, device):
    model.train()
    total_loss = 0
    correct_exp, correct_icm, correct_te = 0, 0, 0
    total = 0
    
    crit_exp, crit_icm, crit_te = criteria
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, l_exp, l_icm, l_te in pbar:
        images = images.to(device)
        l_exp = l_exp.to(device, dtype=torch.long)
        l_icm = l_icm.to(device, dtype=torch.long)
        l_te = l_te.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        out_exp, out_icm, out_te = model(images)
        
        loss_exp = crit_exp(out_exp, l_exp)
        loss_icm = crit_icm(out_icm, l_icm)
        loss_te = crit_te(out_te, l_te)
        
        # Multi-task loss combination
        loss = loss_exp + loss_icm + loss_te
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        
        correct_exp += (torch.argmax(out_exp, dim=1) == l_exp).sum().item()
        correct_icm += (torch.argmax(out_icm, dim=1) == l_icm).sum().item()
        correct_te += (torch.argmax(out_te, dim=1) == l_te).sum().item()
        total += l_exp.size(0)
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    acc_exp = 100 * correct_exp / total
    acc_icm = 100 * correct_icm / total
    acc_te = 100 * correct_te / total
    
    return total_loss / total, (acc_exp, acc_icm, acc_te)

def validate(model, loader, criteria, device):
    model.eval()
    total_loss = 0
    correct_exp, correct_icm, correct_te = 0, 0, 0
    total = 0
    
    crit_exp, crit_icm, crit_te = criteria
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, l_exp, l_icm, l_te in pbar:
            images = images.to(device)
            l_exp = l_exp.to(device, dtype=torch.long)
            l_icm = l_icm.to(device, dtype=torch.long)
            l_te = l_te.to(device, dtype=torch.long)
            
            out_exp, out_icm, out_te = model(images)
            
            loss_exp = crit_exp(out_exp, l_exp)
            loss_icm = crit_icm(out_icm, l_icm)
            loss_te = crit_te(out_te, l_te)
            
            loss = loss_exp + loss_icm + loss_te
            total_loss += loss.item() * images.size(0)
            
            correct_exp += (torch.argmax(out_exp, dim=1) == l_exp).sum().item()
            correct_icm += (torch.argmax(out_icm, dim=1) == l_icm).sum().item()
            correct_te += (torch.argmax(out_te, dim=1) == l_te).sum().item()
            total += l_exp.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
    acc_exp = 100 * correct_exp / total
    acc_icm = 100 * correct_icm / total
    acc_te = 100 * correct_te / total
    
    return total_loss / total, (acc_exp, acc_icm, acc_te)

# ============================================================
# MAIN TRAINING LOOP
# ============================================================
def train_single_model(seed, train_loader, val_loader, w_exp, w_icm, w_te):
    print(f"\n{'='*70}")
    print(f"TRAINING MULTI-TASK MODEL (SEED {seed})")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    model = MultiTaskSwinWithUncertainty(DROPOUT_RATE).to(DEVICE)
    
    # PyTorch's native stable loss function applied to all 3 tasks
    criteria = (
        nn.CrossEntropyLoss(weight=w_exp.to(DEVICE)),
        nn.CrossEntropyLoss(weight=w_icm.to(DEVICE)),
        nn.CrossEntropyLoss(weight=w_te.to(DEVICE))
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_avg_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        t_loss, (t_exp, t_icm, t_te) = train_epoch(model, train_loader, criteria, optimizer, DEVICE)
        v_loss, (v_exp, v_icm, v_te) = validate(model, val_loader, criteria, DEVICE)
        
        # We track the average validation accuracy across all three tasks to determine the "best" model
        avg_val_acc = (v_exp + v_icm + v_te) / 3.0
        
        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {t_loss:.4f} | Acc: EXP={t_exp:.1f}%, ICM={t_icm:.1f}%, TE={t_te:.1f}%")
        print(f"  Val Loss:   {v_loss:.4f} | Acc: EXP={v_exp:.1f}%, ICM={v_icm:.1f}%, TE={v_te:.1f}%")
        print(f"  --> Average Val Acc: {avg_val_acc:.2f}%")
        
        if avg_val_acc > best_avg_acc:
            best_avg_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"multitask_seed{seed}_best.pth"))
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
    
    # Get the sample weights BEFORE we split the dataset
    sample_weights = full_dataset.get_sample_weights()
    
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # We need to manually split the indices to keep track of which weights belong to the training set
    indices = list(range(len(full_dataset)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Extract just the training weights
    train_weights = sample_weights[train_indices]
    
    # Create the sampler! (replacement=True means it can pick the same rare image multiple times per epoch)
    sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)
    
    # Create the Subsets
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    val_ds = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    
    # 🚨 CRITICAL: When using a sampler, shuffle MUST be False 🚨
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    w_exp, w_icm, w_te = full_dataset.get_class_weights()
    
    train_single_model(SEEDS[0], train_loader, val_loader, w_exp, w_icm, w_te)

if __name__ == "__main__":
    main()
