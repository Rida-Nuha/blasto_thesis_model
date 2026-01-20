"""
ICM PERFORMANCE METRICS - FULLY FIXED
✅ Correct paths ✓ Error handling ✓ All metrics
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

# FIXED PATHS - Find your actual ICM models first
print("🔍 FINDING ICM MODEL PATHS...")
possible_paths = [
    "/kaggle/working/blasto_thesis_model/saved_models/uncertainty_ICM/",
    "/kaggle/working/saved_models/uncertainty_ICM/",
    "/kaggle/input/notebook7818469fd6/blasto_thesis_model/saved_models/uncertainty_ICM/"
]

ICM_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        ICM_PATH = path
        print(f"✅ ICM PATH FOUND: {ICM_PATH}")
        break

if ICM_PATH is None:
    print("❌ ICM PATHS NOT FOUND. Available .pth files:")
    for root, dirs, files in os.walk("/kaggle/working"):
        pth_files = [f for f in files if f.endswith('.pth') and 'ICM' in f.upper()]
        if pth_files:
            print(f"📁 {root}:")
            for f in pth_files:
                print(f"   {f}")
    exit()

TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
    cm = confusion_matrix(labels, preds)
    poor_recall = cm[0,0] / (cm[0].sum() + 1e-10) * 100
    good_recall = cm[1,1] / (cm[1].sum() + 1e-10) * 100
    return {
        'accuracy': acc, 'precision': prec*100, 'recall': rec*100, 'macro_f1': f1*100,
        'roc_auc': auc, 'poor_recall': poor_recall, 'good_recall': good_recall, 'confusion_matrix': cm
    }

# FIND ACTUAL ICM MODEL FILES
print(f"\n📂 Scanning {ICM_PATH} for ICM models...")
icm_files = [f for f in os.listdir(ICM_PATH) if f.endswith('.pth') and 'ICM' in f.upper()]
print(f"Found {len(icm_files)} ICM models:")
for f in sorted(icm_files):
    print(f"  ✓ {f}")

icm_models = [f for f in icm_files if any(seed in f for seed in ['42', '123', '456', '789', '2024'])]

# MAIN EXECUTION
val_df = get_icm_full_dataset()
val_dataset = ICMValDataset(val_df, IMG_FOLDER)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

print("\n🔍 ICM INDIVIDUAL MODEL PERFORMANCE")
print("="*70)

individual_results = {}
for model_file in icm_models[:5]:  # Limit to 5 models
    model_path = os.path.join(ICM_PATH, model_file)
    if os.path.exists(model_path):
        print(f"\nEvaluating {model_file}...")
        try:
            model = SwinEmbryoClassifier().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            
            all_preds, all_probs, all_labels = [], [], []
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Eval {model_file[:20]}"):
                    images = images.to(device)
                    outputs = model(images)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs[:, 1].cpu().numpy())
                    all_labels.extend(labels.numpy())
            
            metrics = compute_full_metrics(np.array(all_preds), np.array(all_probs), np.array(all_labels))
            individual_results[model_file] = metrics
            print(f"✅ {model_file}: Acc={metrics['accuracy']:.2f}% | F1={metrics['macro_f1']:.2f}% | AUC={metrics['roc_auc']:.3f}")
        except Exception as e:
            print(f"❌ Error with {model_file}: {e}")
    else:
        print(f"❌ Missing: {model_path}")

# ICM ENSEMBLE (Safe handling)
print("\n" + "="*70)
print("🎯 ICM ENSEMBLE EVALUATION")
print("="*70)

models = []
for model_file in icm_models[:5]:
    model_path = os.path.join(ICM_PATH, model_file)
    if os.path.exists(model_path):
        try:
            model = SwinEmbryoClassifier().to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            model.eval()
            models.append(model)
            print(f"✓ Loaded {model_file}")
        except:
            continue

if len(models) > 0:
    print(f"\nEvaluating {len(models)}-model ensemble...")
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
    
    print(f"\n🏆 ICM {len(models)}-MODEL ENSEMBLE:")
    print(f"   Accuracy:      {ensemble_metrics['accuracy']:.2f}%")
    print(f"   Macro F1:      {ensemble_metrics['macro_f1']:.2f}%")
    print(f"   ROC-AUC:       {ensemble_metrics['roc_auc']:.3f}")
    print(f"   Poor Recall:   {ensemble_metrics['poor_recall']:.1f}%")
    print(f"   Good Recall:   {ensemble_metrics['good_recall']:.1f}%")
    
    # Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(ensemble_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'ICM Ensemble (n={len(models)})\nAcc={ensemble_metrics["accuracy"]:.1f}%')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig('/kaggle/working/icm_ensemble_cm.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save ensemble
    df_ensemble = pd.DataFrame([ensemble_metrics]).round(2)
    df_ensemble.to_csv('/kaggle/working/icm_ensemble_metrics.csv', index=False)
else:
    print("❌ No ICM models could be loaded for ensemble")
    df_ensemble = pd.DataFrame()

# Save individual results
if individual_results:
    df_individual = pd.DataFrame(individual_results).T.round(2)
    df_individual.to_csv('/kaggle/working/icm_individual_metrics.csv')
    print(f"\n✅ SAVED:")
    print(f"📊 /kaggle/working/icm_individual_metrics.csv ({len(individual_results)} models)")
print(f"📊 /kaggle/working/icm_ensemble_metrics.csv")
print(f"🖼️  /kaggle/working/icm_ensemble_cm.png")
