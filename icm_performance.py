"""
ICM INDIVIDUAL + ENSEMBLE PERFORMANCE METRICS
Exact same structure as your TE evaluation
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
import warnings
warnings.filterwarnings('ignore')

# PATHS
ICM_PATH = "/kaggle/working/blasto_thesis_model/saved_models/uncertainty_ICM/"
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

# ICM VALIDATION DATASET (Full dataset - no split issues)
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

# EVALUATE SINGLE MODEL
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
    
    acc = accuracy_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else np.nan
    return {'accuracy': acc, 'roc_auc': auc}

# MAIN EVALUATION
print("🔍 ICM INDIVIDUAL MODEL PERFORMANCE")
print("="*60)

# Create validation data
val_df = get_icm_full_dataset()
val_dataset = ICMValDataset(val_df, IMG_FOLDER)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# Evaluate each ICM model
icm_models = [
    "ICM_silver_seed42_best.pth",
    "ICM_silver_seed123_best.pth",
    "ICM_silver_seed456_best.pth", 
    "ICM_silver_seed789_best.pth",
    "ICM_silver_seed2024_best.pth"
]

results = {}
for model_file in icm_models:
    model_path = os.path.join(ICM_PATH, model_file)
    if os.path.exists(model_path):
        metrics = evaluate_single_model(model_path, val_loader, model_file)
        results[model_file] = metrics
        print(f"✅ {model_file}: {metrics['accuracy']:.2f}% (AUC: {metrics['roc_auc']:.3f})")
    else:
        print(f"❌ Missing: {model_file}")

# SUMMARY TABLE
print("\n" + "="*60)
print("📊 ICM PERFORMANCE SUMMARY")
print("="*60)
df_results = pd.DataFrame(results).T
df_results['accuracy'] = df_results['accuracy'].round(2)
df_results['roc_auc'] = df_results['roc_auc'].round(3)
print(df_results)

print(f"\nEnsemble Average Accuracy: {df_results['accuracy'].mean():.2f}% ± {df_results['accuracy'].std():.2f}%")
print(f"Best Model: {df_results['accuracy'].idxmax()} ({df_results['accuracy'].max():.2f}%)")
print(f"Ensemble AUC: {df_results['roc_auc'].mean():.3f}")

# ENSEMBLE EVALUATION
print("\n" + "="*60)
print("🎯 5-MODEL ICM ENSEMBLE")
print("="*60)

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
    
    # Full metrics
    acc = accuracy_score(all_labels, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"\n🏆 ICM 5-MODEL ENSEMBLE:")
    print(f"   Accuracy:     {acc:.2f}%")
    print(f"   Macro F1:     {f1*100:.2f}%")
    print(f"   ROC-AUC:      {auc:.3f}")
    print(f"   Poor Recall:  {cm[0,0]/cm[0].sum()*100:.1f}%")
    print(f"   Good Recall:  {cm[1,1]/cm[1].sum()*100:.1f}%")
    
    # Confusion Matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'ICM Ensemble Confusion Matrix\nAcc={acc:.1f}%')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.savefig('/kaggle/working/icm_ensemble_cm.png', dpi=300, bbox_inches='tight')
    plt.show()

# Save results
df_results.to_csv('/kaggle/working/icm_individual_results.csv')
ensemble_results = pd.DataFrame({
    'metric': ['accuracy', 'macro_f1', 'roc_auc', 'n_models', 'n_samples'],
    'value': [acc, f1*100, auc, len(models), len(all_labels)]
})
ensemble_results.to_csv('/kaggle/working/icm_ensemble_results.csv', index=False)

print(f"\n💾 SAVED:")
print(f"📊 /kaggle/working/icm_individual_results.csv")
print(f"📊 /kaggle/working/icm_ensemble_results.csv")
print(f"🖼️  /kaggle/working/icm_ensemble_cm.png")
