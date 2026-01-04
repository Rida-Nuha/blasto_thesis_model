"""
ICM (Inner Cell Mass) Quality Prediction with Uncertainty Quantification
Swin Transformer + Deep Ensembles + MC Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
TARGET_SCORE = "ICM_silver"
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "saved_models/uncertainty_ICM"
OUTPUT_DIR = "uncertainty_results_ICM"
BATCH_SIZE = 32
NUM_EPOCHS = 60  # ← Longer for harder task
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
SEEDS = [42, 123, 456, 789, 2024]
PATIENCE = 20  # ← More patience for harder task
MC_DROPOUT_SAMPLES = 50

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# DATASET
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
# LOAD DATA
# ============================================================
print(f"\nLoading dataset for {TARGET_SCORE}...")
df = pd.read_csv(TRAIN_CSV, sep=';')

# Check unique values
print(f"Unique {TARGET_SCORE} values: {sorted(df[TARGET_SCORE].unique())}")

# Binary conversion: ICM >= 2 is Good (adjust based on distribution analysis)
df['label'] = df[TARGET_SCORE].apply(lambda x: 1 if x >= 2 else 0)

# Train-validation split
train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df['label']
)

print(f"\nDataset Split:")
print(f"  Total samples: {len(df)}")
print(f"  Training: {len(train_df)}")
print(f"  Validation: {len(val_df)}")

print(f"\nClass Distribution:")
poor_count = (df['label']==0).sum()
good_count = (df['label']==1).sum()
print(f"  Poor (ICM < 2): {poor_count} ({poor_count/len(df)*100:.1f}%)")
print(f"  Good (ICM >= 2): {good_count} ({good_count/len(df)*100:.1f}%)")

# Compute class weights with extra boost for minority
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = len(train_df) / (2 * class_counts)

# If very imbalanced, boost minority class more
if min(class_counts) / max(class_counts) < 0.25:  # If minority < 25%
    minority_idx = class_counts.argmin()
    class_weights[minority_idx] *= 1.5  # Boost by 50%
    print(f"\n⚠ High imbalance detected! Boosting minority class weight.")

class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\nClass weights: {class_weights.cpu().numpy()}")

# ============================================================
# TRANSFORMS (More aggressive for ICM)
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(25),  # ← More rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # ← More augmentation
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ← Add shift
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# MODEL (Same as TE)
# ============================================================
class SwinEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4):  # ← Higher dropout for ICM
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
# FOCAL LOSS
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.5):  # ← Higher gamma for harder task
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(seed):
    print(f"\n{'='*70}")
    print(f"TRAINING MODEL WITH SEED {seed}")
    print(f"{'='*70}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Data loaders
    train_dataset = EmbryoDataset(train_df, IMG_FOLDER, train_transform)
    val_dataset = EmbryoDataset(val_df, IMG_FOLDER, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Model, loss, optimizer
    model = SwinEmbryoClassifier().to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_acc = 0.0
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        # ============ TRAINING ============
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # ============ VALIDATION ============
        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        # ============ SAVE BEST MODEL ============
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            save_path = f"{MODEL_DIR}/ICM_silver_seed{seed}_best.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved! Accuracy: {best_acc*100:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        # ============ EARLY STOPPING ============
        if patience_counter >= PATIENCE:
            print(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    print(f"\n✅ Training complete for seed {seed}")
    print(f"   Best validation accuracy: {best_acc*100:.2f}%")
    return best_acc

# ============================================================
# TRAIN ALL 5 MODELS
# ============================================================
print("\n" + "="*70)
print(f"STARTING 5-MODEL ENSEMBLE TRAINING FOR {TARGET_SCORE}")
print("="*70)

results = {}
for seed in SEEDS:
    best_acc = train_model(seed)
    results[seed] = best_acc

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

for seed, acc in results.items():
    print(f"Seed {seed:4d}: {acc*100:.2f}%")

avg_acc = np.mean(list(results.values()))
print(f"\nAverage accuracy across 5 models: {avg_acc*100:.2f}%")
print(f"Paper's ICM accuracy: 63.69%")
improvement = (avg_acc - 0.6369) * 100
print(f"Your improvement: {improvement:+.2f}%")

print(f"\n✅ All models saved in: {MODEL_DIR}/")
print(f"✅ Ready for evaluation!")
