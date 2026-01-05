"""
confu_icm_check.py - Standalone ICM Confusion Matrix Check
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import os

# ====== DEVICE ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== PATHS ======
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "saved_models/uncertainty_ICM"

# ====== DATASET CLASS ======
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

# ====== MODEL CLASS ======
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

# ====== LOAD DATA ======
print("Loading data...")
df = pd.read_csv(TRAIN_CSV, sep=';')
df['label'] = df['ICM_silver'].apply(lambda x: 1 if x >= 2 else 0)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=42, stratify=df['label']
)

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_dataset = EmbryoDataset(val_df, IMG_FOLDER, val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ====== LOAD MODELS ======
print("\nLoading trained ICM models...")

models = []
SEEDS = [42, 123, 456, 789, 2024]

for seed in SEEDS:
    model = SwinEmbryoClassifier().to(device)
    model_path = f"{MODEL_DIR}/ICM_silver_seed{seed}_best.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    models.append(model)
    print(f"✓ Loaded seed {seed}")

# ====== GET PREDICTIONS ======
print("\nGetting predictions...")

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        
        batch_outputs = []
        for model in models:
            outputs = model(images)
            batch_outputs.append(outputs)
        
        avg_outputs = torch.stack(batch_outputs).mean(dim=0)
        _, preds = torch.max(avg_outputs, 1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

val_preds = np.array(all_predictions)
val_labels = np.array(all_labels)

# ====== ANALYSIS ======
print("\n" + "="*70)
print("ICM EVALUATION RESULTS")
print("="*70)

accuracy = accuracy_score(val_labels, val_preds)
print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(val_labels, val_preds)
print(f"\nConfusion Matrix:")
print("                 Predicted")
print("                 Poor  Good")
print(f"Actual Poor     {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Good     {cm[1,0]:4d}  {cm[1,1]:4d}")

poor_recall = cm[0,0] / (cm[0,0] + cm[0,1])
good_recall = cm[1,1] / (cm[1,0] + cm[1,1])

print(f"\nPer-Class Recall:")
print(f"  Poor: {poor_recall*100:.1f}%")
print(f"  Good: {good_recall*100:.1f}%")

print(f"\nClassification Report:")
print(classification_report(val_labels, val_preds, 
                          target_names=['Poor', 'Good'],
                          digits=4))

if good_recall > 0.80 and poor_recall > 0.80:
    print("\n✅ REAL PERFORMANCE! Both classes learned well!")
else:
    print("\n⚠️ Check if model is biased")
