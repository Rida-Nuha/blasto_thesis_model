"""
REAL MODEL EVALUATION - YOUR EXACT MODEL PATHS
ICM/TE/EXP 5-model ensembles → Springer Table 1 + Figures
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
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# YOUR EXACT MODEL PATHS - CORRECTED!
# ============================================================
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"

# YOUR ACTUAL MODEL DIRECTORIES
MODEL_PATHS = {
    'icm': "/kaggle/working/blasto_thesis_model/saved_models/uncertainty_ICM",
    'te':  "/kaggle/working/blasto_thesis_model/saved_models/uncertainty_TE",
    'exp': "/kaggle/working/saved_models/uncertainty"
}

TARGETS = ['ICM_silver', 'TE_silver', 'EXP_silver']
SEEDS = [42, 123, 456, 789, 2024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Transforms
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# MODEL CLASS (Matches your training)
# ============================================================
class SwinEmbryoClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================
# DATASET CLASS
# ============================================================
class TrainValDataset(Dataset):
    def __init__(self, csv_file, img_folder, target_col, threshold=2, transform=None, split='val', seed=42):
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.target_col = target_col
        self.threshold = threshold
        self.transform = transform
        
        # Filter valid samples
        valid_mask = (
            self.df[target_col].notna() & 
            (self.df[target_col] != 'ND') & 
            (self.df[target_col] != 'NA')
        )
        self.df = self.df[valid_mask].copy()
        self.df[target_col] = pd.to_numeric(self.df[target_col], errors='coerce')
        self.df = self.df[self.df[target_col].notna()].copy()
        
        # Binary labels (≥2 = Good)
        self.df['label'] = (self.df[target_col] >= threshold).astype(int)
        
        # Validation split (15% stratified)
        if split == 'val':
            train_df = self.df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(frac=0.85, random_state=seed)
            )
            self.df = self.df.drop(train_df.index).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row['label'], dtype=torch.long)

# ============================================================
# EVALUATION FUNCTIONS
# ============================================================
def load_models(model_dir, target_key):
    """Load all 5 seed models for a target"""
    models = []
    for seed in SEEDS:
        model_path = f"{model_dir}/{target_key.upper()}_silver_seed{seed}_best.pth"
        if os.path.exists(model_path):
            model = SwinEmbryoClassifier().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
            print(f"✓ Loaded {os.path.basename(model_path)}")
        else:
            print(f"⚠️ Missing: {model_path}")
    return models

def evaluate_ensemble(models, loader, target_name):
    """Ensemble evaluation"""
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Eval {target_name}"):
            images = images.to(device)
            
            # 5-model ensemble
            batch_outputs = [model(images) for model in models]
            avg_outputs = torch.stack(batch_outputs).mean(dim=0)
            probs = torch.softmax(avg_outputs, dim=1)
            preds = torch.argmax(avg_outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Good class prob
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def compute_metrics(preds, probs, labels):
    """Complete metrics"""
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else np.nan
    
    cm = confusion_matrix(labels, preds)
    poor_recall = cm[0,0] / (cm[0].sum() + 1e-10)
    good_recall = cm[1,1] / (cm[1].sum() + 1e-10)
    
    return {
        'accuracy': acc * 100,
        'macro_f1': f1 * 100,
        'poor_recall': poor_recall * 100,
        'good_recall': good_recall * 100,
        'roc_auc': auc,
        'confusion_matrix': cm
    }

# ============================================================
# MAIN EVALUATION
# ============================================================
print("\n🚀 LOADING YOUR 15 MODELS (5×ICM + 5×TE + 5×EXP)")
results = {}

for target_col, target_key in zip(TARGETS, ['icm', 'te', 'exp']):
    print(f"\n{'='*60}")
    print(f"🎯 EVALUATING {target_key.upper()}")
    print('='*60)
    
    # Create validation dataset
    val_dataset = TrainValDataset(TRAIN_CSV, IMG_FOLDER, target_col, split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"📊 Val samples: {len(val_dataset)}")
    print(f"   Dist: {val_dataset.df['label'].value_counts(normalize=True).round(3)}")
    
    # Load YOUR models
    model_dir = MODEL_PATHS[target_key]
    models = load_models(model_dir, target_key)
    
    if len(models) == 0:
        print(f"❌ No models loaded for {target_key}")
        continue
    
    # Evaluate ensemble
    preds, probs, labels = evaluate_ensemble(models, val_loader, target_key)
    metrics = compute_metrics(preds, probs, labels)
    
    results[target_key.upper()] = {
        **metrics,
        'n_models': len(models),
        'n_samples': len(labels)
    }
    
    print(f"\n✅ {target_key.upper()} RESULTS:")
    print(f"   Accuracy: {metrics['accuracy']:.2f}%")
    print(f"   Macro F1: {metrics['macro_f1']:.2f}%")
    print(f"   ROC-AUC:  {metrics['roc_auc']:.3f}")
    print(f"   Poor Rec: {metrics['poor_recall']:.2f}%")
    print(f"   Good Rec: {metrics['good_recall']:.2f}%")

# ============================================================
# SPRINGER TABLE 1 & FIGURES
# ============================================================
if results:
    print("\n" + "="*80)
    print("🏆 SPRINGER TABLE 1: YOUR REAL RESULTS")
    print("="*80)
    
    df_results = pd.DataFrame(results).T
    display_cols = ['accuracy', 'macro_f1', 'poor_recall', 'good_recall', 'roc_auc']
    df_display = df_results[display_cols].round(2)
    
    print(df_display)
    df_display.to_csv('/kaggle/working/springer_table1_real_results.csv')
    
    # Figure 1: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Figure 1: Validation Confusion Matrices (5-Model Ensembles)', fontsize=16, fontweight='bold')
    
    for i, target in enumerate(results.keys()):
        cm = results[target]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, 
                   square=True, linewidths=2)
        axes[i].set_title(f'{target}\nAcc={results[target]["accuracy"]:.1f}%', fontweight='bold')
        axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('/kaggle/working/figure1_real_confusion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ FILES SAVED:")
    print("📊 /kaggle/working/springer_table1_real_results.csv")
    print("🖼️  /kaggle/working/figure1_real_confusion.png")
    print("🎉 YOUR REAL SPRINGER RESULTS READY!")

else:
    print("❌ No results - check model paths")
