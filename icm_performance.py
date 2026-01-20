"""
ICM PERFORMANCE METRICS - COMPLETE (Fixed import os)
Generates: Accuracy, ROC-AUC, F1, Precision, Recall, Confusion Matrix
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
import os  # ← FIXED! Missing import
import warnings
warnings.filterwarnings('ignore')

# PATHS
ICM_PATH = "/kaggle/working/blasto_thesis_model/saved_models/uncertainty_ICM/"
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ALL METRICS GENERATED:
print("✅ Metrics Generated: Accuracy, ROC-AUC, Precision, Recall, F1, Confusion Matrix")
print("📊 Individual + 5-Model Ensemble Results")

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

# ICM FULL DATASET (2044 samples)
def get_icm_full_dataset():
    df = pd.read_csv(TRAIN_CSV, sep=';')
    valid_mask = df['ICM_silver'].notna() & (df['ICM_silver'] != 'ND') & (df['ICM_silver'] != 'NA')
    df = df[valid_mask].copy()
    df['ICM_silver'] = pd.to_numeric(df['ICM_silver'], errors='coerce')
    df = df[df['ICM_silver'].notna()].copy()
    df['label'] = (df['ICM_silver'] >= 2).astype(int)
    print(f"✅ ICM Full Dataset: {len(df)} samples")
    print(f"   Poor/Good: {df['label'].value_counts().to_dict()}")
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

# COMPLETE METRICS FUNCTION
def compute_full_metrics(preds, probs, labels):
    acc = accuracy_score(labels, preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
    cm = confusion_matrix(labels, preds)
    
    poor_recall = cm[0,0] / (cm[0].sum() + 1e-10) * 100
    good_recall = cm[1,1] / (cm[1].sum() + 1e-10) * 100
    
    return {
        'accuracy': acc,
        'precision': prec*100, 
        'recall': rec*100,
        'macro_f1': f1*100,
        'roc_auc': auc,
        'poor_recall': poor_recall,
        'good_recall': good_recall,
        'confusion_matrix': cm
    }

# EVALUATE SINGLE MODEL (ALL METRICS)
def evaluate_single_model(model_path, val_loader, model_name):
    model = SwinEmbryoClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Eval {model_name}"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return compute_full_metrics(np.array(all_preds), np.array(all_probs), np.array(all_labels))

# MAIN EXECUTION
print("\n🔍 ICM INDIVIDUAL MODEL PERFORMANCE")
print("="*70)

val_df = get_icm_full_dataset()
val_dataset = ICMValDataset(val_df, IMG_FOLDER)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

icm_models = [
    "ICM_silver_seed42_best.pth",
    "ICM_silver_seed123_best.pth", 
    "ICM_silver_seed456_best.pth",
    "ICM_silver_seed789_best.pth",
    "ICM_silver_seed2024_best.pth"
]

individual_results = {}
for model_file in icm_models:
    model_path = os.path.join(ICM_PATH, model_file)  # Now works!
    if os.path.exists(model_path):
        metrics = evaluate_single_model(model_path, val_loader, model_file)
        individual_results[model_file] = metrics
        print(f"✅ {model_file}:")
        print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['macro_f1']:.2f}% | AUC: {metrics['roc_auc']:.3f}")
    else:
        print(f"❌ Missing: {model_path}")

# ICM 5-MODEL ENSEMBLE (ALL METRICS)
print("\n" + "="*70)
print("🎯 ICM 5-MODEL ENSEMBLE (ALL METRICS)")
print("="*70)

models = []
for model_file in icm_models:
    model_path = os.path.join(ICM_PATH, model_file)
    if os.path.exists(model_path):
        model = SwinEmbryoClassifier().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        model.eval()
        models.append(model)

if len(models) > 0:
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
    
    print(f"\n🏆 ICM ENSEMBLE COMPLETE METRICS:")
    print(f"   Accuracy:      {ensemble_metrics['accuracy']:.2f}%")
    print(f"   Precision:     {ensemble_metrics['precision']:.2f}%")
    print(f"   Recall:        {ensemble_metrics['recall']:.2f}%")
    print(f"   Macro F1:      {ensemble_metrics['macro_f1']:.2f}%")
    print(f"   ROC-AUC:       {ensemble_metrics['roc_auc']:.3f}")
    print(f"   Poor Recall:   {ensemble_metrics['poor_recall']:.1f}%")
    print(f"   Good Recall:   {ensemble_metrics['good_recall']:.1f}%")
    
    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(ensemble_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                cbar=True, square=True)
    plt.title(f'ICM 5-Model Ensemble Confusion Matrix\nAcc={ensemble_metrics["accuracy"]:.1f}%')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig('/kaggle/working/icm_ensemble_cm.png', dpi=300, bbox_inches='tight')
    plt.show()

# SAVE ALL RESULTS
df_individual = pd.DataFrame(individual_results).T.round(2)
df_ensemble = pd.DataFrame([ensemble_metrics]).round(2)

df_individual.to_csv('/kaggle/working/icm_individual_metrics.csv')
df_ensemble.to_csv('/kaggle/working/icm_ensemble_metrics.csv')

print(f"\n💾 SAVED FILES:")
print(f"📊 /kaggle/working/icm_individual_metrics.csv")
print(f"📊 /kaggle/working/icm_ensemble_metrics.csv") 
print(f"🖼️  /kaggle/working/icm_ensemble_cm.png")
print("\n✅ ALL METRICS GENERATED!")
