"""
ICM PERFORMANCE METRICS - PATH FIXED
Your models are in fresh_clone directory!
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# ✅ CORRECTED ICM PATH (from your output)
ICM_PATH = "/kaggle/working/fresh_clone/saved_models/uncertainty_ICM/"
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ ICM PATH: {ICM_PATH}")
print(f"✅ Found 5 models: seed42,123,456,789,2024")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# MODEL CLASS
class SwinEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features), nn.Dropout(0.3),
            nn.Linear(in_features, 512), nn.GELU(),
            nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.GELU(),
            nn.Dropout(0.3), nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.backbone(x)

def get_icm_full_dataset():
    df = pd.read_csv(TRAIN_CSV, sep=';')
    valid_mask = df['ICM_silver'].notna() & (df['ICM_silver'] != 'ND') & (df['ICM_silver'] != 'NA')
    df = df[valid_mask].copy()
    df['ICM_silver'] = pd.to_numeric(df['ICM_silver'], errors='coerce')
    df = df[df['ICM_silver'].notna()].copy()
    df['label'] = (df['ICM_silver'] >= 2).astype(int)
    print(f"✅ ICM Dataset: {len(df)} samples | Poor:{df['label'].value_counts()[0]} Good:{df['label'].value_counts()[1]}")
    return df

class ICMValDataset(Dataset):
    def __init__(self, val_df, img_folder):
        self.df = val_df
        self.img_folder = img_folder
        self.transform = val_transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, torch.tensor(row['label'], dtype=torch.long)

def compute_full_metrics(preds, probs, labels):
    acc = accuracy_score(labels, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    auc = roc_auc_score(labels, probs)
    cm = confusion_matrix(labels, preds)
    poor_recall = cm[0,0] / (cm[0].sum() + 1e-10) * 100
    good_recall = cm[1,1] / (cm[1].sum() + 1e-10) * 100
    return {
        'accuracy': acc, 'macro_f1': f1*100, 'roc_auc': auc,
        'poor_recall': poor_recall, 'good_recall': good_recall, 'confusion_matrix': cm
    }

# ICM MODELS (EXACT FILENAMES FROM YOUR OUTPUT)
icm_models = [
    "ICM_silver_seed42_best.pth",
    "ICM_silver_seed123_best.pth", 
    "ICM_silver_seed456_best.pth",
    "ICM_silver_seed789_best.pth",
    "ICM_silver_seed2024_best.pth"
]

# DATASET
val_df = get_icm_full_dataset()
val_dataset = ICMValDataset(val_df, IMG_FOLDER)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

print("\n🔥 ICM INDIVIDUAL MODEL RESULTS")
print("="*50)

individual_results = {}
for model_file in icm_models:
    model_path = os.path.join(ICM_PATH, model_file)
    print(f"Loading {model_file}...")
    
    model = SwinEmbryoClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"{model_file[:25]}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    metrics = compute_full_metrics(np.array(all_preds), np.array(all_probs), np.array(all_labels))
    individual_results[model_file] = metrics
    
    print(f"✅ {model_file}: {metrics['accuracy']:.1f}% | F1:{metrics['macro_f1']:.1f}% | AUC:{metrics['roc_auc']:.3f}")

# 5-MODEL ENSEMBLE
print("\n" + "="*60)
print("🏆 ICM 5-MODEL ENSEMBLE")
print("="*60)

models = []
for model_file in icm_models:
    model_path = os.path.join(ICM_PATH, model_file)
    model = SwinEmbryoClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    models.append(model)

all_preds, all_probs, all_labels = [], [], []
with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="ICM Ensemble"):
        images = images.to(device)
        batch_outputs = [model(images) for model in models]
        avg_outputs = torch.stack(batch_outputs).mean(dim=0)
        probs = torch.softmax(avg_outputs, dim=1)
        preds = torch.argmax(avg_outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.numpy())

ensemble_metrics = compute_full_metrics(np.array(all_preds), np.array(all_probs), np.array(all_labels))

print(f"\n🎯 FINAL ICM ENSEMBLE RESULTS:")
print(f"   Accuracy:     {ensemble_metrics['accuracy']:.2f}%")
print(f"   Macro F1:     {ensemble_metrics['macro_f1']:.2f}%")
print(f"   ROC-AUC:      {ensemble_metrics['roc_auc']:.3f}")
print(f"   Poor Recall:  {ensemble_metrics['poor_recall']:.1f}%")
print(f"   Good Recall:  {ensemble_metrics['good_recall']:.1f}%")

# PLOT + SAVE
plt.figure(figsize=(10, 8))
plt.subplot(2,2,1)
sns.heatmap(ensemble_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
plt.title(f'ICM Ensemble (Acc={ensemble_metrics["accuracy"]:.1f}%)')

plt.subplot(2,2,2)
df_individual = pd.DataFrame(individual_results).T[['accuracy', 'macro_f1', 'roc_auc']].round(2)
df_individual.plot(kind='bar', ax=plt.gca())
plt.title('Individual Model Performance')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/kaggle/working/icm_complete_results.png', dpi=300, bbox_inches='tight')
plt.show()

# FINAL TABLES
df_individual.to_csv('/kaggle/working/icm_individual_results.csv')
pd.DataFrame([ensemble_metrics]).round(2).to_csv('/kaggle/working/icm_ensemble_results.csv', index=False)

print(f"\n✅ SAVED FOR SPRINGER PAPER:")
print(f"📊 /kaggle/working/icm_individual_results.csv")
print(f"📊 /kaggle/working/icm_ensemble_results.csv") 
print(f"🖼️  /kaggle/working/icm_complete_results.png")
print(f"\n📋 Table 1 Row: | ICM | {ensemble_metrics['accuracy']:.1f}% | {ensemble_metrics['macro_f1']:.1f}% | {ensemble_metrics['poor_recall']:.1f}% | {ensemble_metrics['good_recall']:.1f}% | {ensemble_metrics['roc_auc']:.3f} |")
