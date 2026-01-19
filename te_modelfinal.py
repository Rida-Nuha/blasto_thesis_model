"""
TE CLASSIFICATION - USING ICM'S PROVEN ARCHITECTURE
Same code that achieved 96.35% for ICM
Only difference: Target = TE_silver instead of ICM_silver
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import random
import os

# ============================================================
# CONFIGURATION - SAME AS ICM
# ============================================================
SEEDS = [42, 123, 456, 789, 2024]  # Train 5 models like ICM
CSV_FILE = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
TARGET = 'TE_silver'  # ← ONLY CHANGE FROM ICM CODE
SAVE_DIR = "saved_models/uncertainty_TE"
os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Device: {device}")

# ============================================================
# SEED FUNCTION
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# DATASET - EXACT SAME AS ICM
# ============================================================
class EmbryoDataset(Dataset):
    def __init__(self, df, img_folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.loc[idx, 'Image']
        label = self.df.loc[idx, 'label']
        
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

# ============================================================
# MODEL - EXACT SAME AS ICM
# ============================================================
class SwinEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================
# LOAD DATA
# ============================================================
print("\n" + "="*70)
print("LOADING TE DATA")
print("="*70)

df = pd.read_csv(CSV_FILE, sep=';')
print(f"✓ Total samples: {len(df)}")

# Filter valid TE values
te_df = df[df[TARGET].notna()].copy()
te_df = te_df[te_df[TARGET] != 'ND'].copy()
te_df = te_df[te_df[TARGET] != 'NA'].copy()
te_df[TARGET] = pd.to_numeric(te_df[TARGET], errors='coerce')
te_df = te_df[te_df[TARGET].notna()].copy()

print(f"✓ Valid TE samples: {len(te_df)}")
print(f"\nTE distribution:")
print(te_df[TARGET].value_counts().sort_index())

# Binary conversion (TE >= 2 is Good) - SAME AS ICM
te_df['label'] = te_df[TARGET].apply(lambda x: 1 if x >= 2 else 0)

poor_count = (te_df['label'] == 0).sum()
good_count = (te_df['label'] == 1).sum()

print(f"\nBinary label distribution:")
print(f"  Poor (TE < 2): {poor_count} ({poor_count/len(te_df)*100:.1f}%)")
print(f"  Good (TE >= 2): {good_count} ({good_count/len(te_df)*100:.1f}%)")

# ============================================================
# TRANSFORMS - EXACT SAME AS ICM
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# TRAINING LOOP FOR EACH SEED
# ============================================================
print("\n" + "="*70)
print("TRAINING TE MODELS WITH 5 SEEDS")
print("="*70)

all_results = []

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*70}")
    print(f"SEED {seed} ({seed_idx}/5)")
    print(f"{'='*70}")
    
    set_seed(seed)
    
    # Split data with current seed
    train_df, val_df = train_test_split(
        te_df, test_size=0.2, stratify=te_df['label'], random_state=seed
    )
    
    print(f"\nTrain: {len(train_df)} | Validation: {len(val_df)}")
    
    # Calculate class weights
    class_counts = train_df['label'].value_counts().sort_index()
    total = len(train_df)
    class_weights = torch.tensor([
        total / (2.0 * class_counts[0]),
        total / (2.0 * class_counts[1])
    ], dtype=torch.float32).to(device)
    
    print(f"Class weights: Poor={class_weights[0]:.3f}, Good={class_weights[1]:.3f}")
    
    # Create datasets
    train_dataset = EmbryoDataset(train_df, IMG_FOLDER, train_transform)
    val_dataset = EmbryoDataset(val_df, IMG_FOLDER, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Initialize model - EXACT SAME AS ICM
    model = SwinEmbryoClassifier(num_classes=2, dropout=0.4).to(device)
    
    # Loss, optimizer, scheduler - EXACT SAME AS ICM
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
    
    # Training - EXACT SAME AS ICM
    NUM_EPOCHS = 25
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step()
        
        # Print summary
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% {'🎯 BEST!' if val_acc > best_val_acc else ''}")
        
        # Confusion matrix every 5 epochs
        if (epoch + 1) % 5 == 0:
            cm = confusion_matrix(val_labels, val_preds)
            print(f"\n  Validation Confusion Matrix:")
            print(f"    Predicted: Poor  Good")
            print(f"    Poor:     {cm[0,0]:4d}  {cm[0,1]:4d}")
            print(f"    Good:     {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{SAVE_DIR}/TE_silver_seed{seed}_best.pth")
            print(f"  ✅ Saved best model!")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    # Final evaluation for this seed
    print(f"\n{'='*70}")
    print(f"SEED {seed} FINAL RESULTS")
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(f"{SAVE_DIR}/TE_silver_seed{seed}_best.pth"))
    model.eval()
    
    val_preds, val_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    final_acc = accuracy_score(val_labels, val_preds)
    cm = confusion_matrix(val_labels, val_preds)
    
    print(f"\n✅ Best Validation Accuracy: {final_acc*100:.2f}%")
    print(f"\nConfusion Matrix:")
    print(f"             Predicted")
    print(f"             Poor  Good")
    print(f"Actual Poor {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Good {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Calculate per-class metrics
    poor_recall = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1])>0 else 0
    good_recall = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1])>0 else 0
    
    print(f"\nPer-Class Recall:")
    print(f"  Poor: {poor_recall*100:.2f}%")
    print(f"  Good: {good_recall*100:.2f}%")
    
    all_results.append({
        'seed': seed,
        'accuracy': final_acc,
        'poor_recall': poor_recall,
        'good_recall': good_recall
    })

# ============================================================
# ENSEMBLE RESULTS
# ============================================================
print("\n" + "="*70)
print("ENSEMBLE RESULTS - ALL 5 TE MODELS")
print("="*70)

for result in all_results:
    print(f"\nSeed {result['seed']}:")
    print(f"  Accuracy:    {result['accuracy']*100:.2f}%")
    print(f"  Poor Recall: {result['poor_recall']*100:.2f}%")
    print(f"  Good Recall: {result['good_recall']*100:.2f}%")

avg_acc = np.mean([r['accuracy'] for r in all_results])
std_acc = np.std([r['accuracy'] for r in all_results])

print(f"\n{'='*70}")
print(f"FINAL TE ENSEMBLE PERFORMANCE")
print(f"{'='*70}")
print(f"Average Accuracy: {avg_acc*100:.2f}% ± {std_acc*100:.2f}%")
print(f"Best Single Model: {max([r['accuracy'] for r in all_results])*100:.2f}%")
print(f"Worst Single Model: {min([r['accuracy'] for r in all_results])*100:.2f}%")

print("\n✅ TE TRAINING COMPLETE!")
print(f"Models saved in: {SAVE_DIR}/")
