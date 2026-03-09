"""
Multi-Task EfficientNet-B0 Baseline
Parameter-Efficient CNN Comparison Architecture for Comprehensive Embryo Grading (EXP, ICM, TE)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
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
SAVE_DIR = "/kaggle/working/saved_models/efficientnet_b0_baseline"        

NUM_CLASSES_EXP = 5  
NUM_CLASSES_ICM = 3  
NUM_CLASSES_TE = 3   

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
# 2. DATASET (Identical Strict 3-Class Filter)
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
        
        print(f"\n{'='*60}")
        print("MULTI-TASK DATASET LOADED (STRICT 3-CLASS FOR ICM/TE)")
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
        
        exp_val = int(row["EXP_silver"])
        l_exp = exp_val - 1 if exp_val >= 1 else exp_val
        l_exp = torch.tensor(l_exp, dtype=torch.long)
        
        l_icm = torch.tensor(int(row["ICM_silver"]), dtype=torch.long)
        l_te = torch.tensor(int(row["TE_silver"]), dtype=torch.long)
        
        return image, l_exp, l_icm, l_te
    
    def get_class_weights(self):
        total = len(self.df)
        def calc_weights(col_name, num_c):
            weights = np.ones(num_c, dtype=np.float32)
            if col_name == "EXP_silver":
                counts = (self.df[col_name] - 1).value_counts()
            else:
                counts = self.df[col_name].value_counts()
            for i in range(num_c):
                if i in counts:
                    weights[i] = total / (num_c * counts[i])
                else:
                    weights[i] = 1.0  
            return torch.FloatTensor(weights)
            
        return calc_weights("EXP_silver", NUM_CLASSES_EXP), calc_weights("ICM_silver", NUM_CLASSES_ICM), calc_weights("TE_silver", NUM_CLASSES_TE)

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
# 3. MULTI-TASK EFFICIENTNET-B0 MODEL
# ============================================================
class MultiTaskEfficientNetB0(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        print("Injecting Standard ImageNet Weights for EfficientNet-B0...")
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # In EfficientNet, the classifier is a Sequential. The Linear layer is at index 1.
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Exact same head architecture as Swin-T and ResNet50
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
# 4. UTILS & TRAINING LOOP
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
    crit_exp, crit_icm, crit_te = criteria
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for images, l_exp, l_icm, l_te in pbar:
        images, l_exp, l_icm, l_te = images.to(device), l_exp.to(device), l_icm.to(device), l_te.to(device)
        
        optimizer.zero_grad()
        out_exp, out_icm, out_te = model(images)
        
        loss = crit_exp(out_exp, l_exp) + crit_icm(out_icm, l_icm) + crit_te(out_te, l_te)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        correct_exp += (torch.argmax(out_exp, dim=1) == l_exp).sum().item()
        correct_icm += (torch.argmax(out_icm, dim=1) == l_icm).sum().item()
        correct_te += (torch.argmax(out_te, dim=1) == l_te).sum().item()
        total += l_exp.size(0)
        
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def validate(model, loader, criteria, device):
    model.eval()
    total_loss, correct_exp, correct_icm, correct_te, total = 0, 0, 0, 0, 0
    crit_exp, crit_icm, crit_te = criteria
    
    with torch.no_grad():
        for images, l_exp, l_icm, l_te in tqdm(loader, desc="Validation", leave=False):
            images, l_exp, l_icm, l_te = images.to(device), l_exp.to(device), l_icm.to(device), l_te.to(device)
            out_exp, out_icm, out_te = model(images)
            
            loss = crit_exp(out_exp, l_exp) + crit_icm(out_icm, l_icm) + crit_te(out_te, l_te)
            total_loss += loss.item() * images.size(0)
            
            correct_exp += (torch.argmax(out_exp, dim=1) == l_exp).sum().item()
            correct_icm += (torch.argmax(out_icm, dim=1) == l_icm).sum().item()
            correct_te += (torch.argmax(out_te, dim=1) == l_te).sum().item()
            total += l_exp.size(0)
            
    return total_loss / total, (100 * correct_exp / total, 100 * correct_icm / total, 100 * correct_te / total)

def train_single_model(seed, train_loader, val_loader, w_exp, w_icm, w_te):
    print(f"\n{'='*70}")
    print(f"TRAINING EFFICIENTNET-B0 BASELINE (SEED {seed})")
    print(f"{'='*70}\n")
    
    set_seed(seed)
    model = MultiTaskEfficientNetB0(DROPOUT_RATE).to(DEVICE)
    
    criteria = (nn.CrossEntropyLoss(weight=w_exp.to(DEVICE)), nn.CrossEntropyLoss(weight=w_icm.to(DEVICE)), nn.CrossEntropyLoss(weight=w_te.to(DEVICE)))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_avg_acc, best_epoch, patience_counter = 0, 0, 0
    
    for epoch in range(EPOCHS):
        t_loss, (t_exp, t_icm, t_te) = train_epoch(model, train_loader, criteria, optimizer, DEVICE)
        v_loss, (v_exp, v_icm, v_te) = validate(model, val_loader, criteria, DEVICE)
        
        avg_val_acc = (v_exp + v_icm + v_te) / 3.0
        
        print(f"Epoch {epoch+1}/{EPOCHS}:  Val Loss: {v_loss:.4f} | Avg Acc: {avg_val_acc:.2f}% (EXP:{v_exp:.1f}%, ICM:{v_icm:.1f}%, TE:{v_te:.1f}%)")
        
        if avg_val_acc > best_avg_acc:
            best_avg_acc, best_epoch, patience_counter = avg_val_acc, epoch + 1, 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"efficientnet_b0_multitask_seed{seed}_best.pth"))
            print(f"   🎯 New best average accuracy! (saved)")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break
        scheduler.step()
        
    print(f"\n✅ Training finished! Best Avg Acc: {best_avg_acc:.2f}% at epoch {best_epoch}")

def main():
    print(f"Device: {DEVICE}")
    full_dataset = MultiTaskGardnerDataset(TRAIN_CSV, IMG_FOLDER, transform=None)
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    train_ds, val_ds = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    w_exp, w_icm, w_te = full_dataset.get_class_weights()
    train_single_model(SEEDS[0], train_loader, val_loader, w_exp, w_icm, w_te)

if __name__ == "__main__":
    main()
