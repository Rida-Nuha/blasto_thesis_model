"""
Training/Validation Performance Metrics - ICM/TE/EXP Ensembles
Generates Springer Table 1 + Figures WITHOUT test data
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
from sklearn.metrics import classification_report

# ============================================================
# CONFIGURATION - TRAINING/VAL ONLY
# ============================================================
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"

# Model directories
MODEL_DIRS = {
    'ICM': "/kaggle/working/saved_models/uncertainty",
    'TE': "/kaggle/working/saved_models/uncertainty", 
    'EXP': "/kaggle/working/saved_models/uncertainty"
}

TARGETS = ['ICM_silver', 'TE_silver', 'EXP_silver']  # TRAINING columns
BINARY_THRESH = 2

SEEDS = [42, 123, 456, 789, 2024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL CLASS
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
# TRAINING DATASET WITH VAL SPLIT
# ============================================================
class TrainValDataset(Dataset):
    def __init__(self, df, img_folder, target_col, threshold=2, transform=None, split='train'):
        self.df = df.copy()
        self.img_folder = img_folder
        self.target_col = target_col
        self.threshold = threshold
        self.transform = transform
        self.split = split
        
        # Filter valid samples
        valid_mask = (
            self.df[target_col].notna() & 
            (self.df[target_col] != 'ND') & 
            (self.df[target_col] != 'NA')
        )
        self.df = self.df[valid_mask].copy()
        self.df[target_col] = pd.to_numeric(self.df[target_col], errors='coerce')
        self.df = self.df[self.df[target_col].notna()].copy()
        
        # Binary labels
        self.df['label'] = (self.df[target_col] >= threshold).astype(int)
        
        if split == 'train':
            self.df = self.df.sample(frac=0.85, random_state=42).reset_index(drop=True)
        else:  # val
            train_idx = self.df.sample(frac=0.85, random_state=42).index
            self.df = self.df.drop(train_idx).reset_index(drop=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, row['label']

# ============================================================
# COMPLETE EVALUATION FUNCTION
# ============================================================
def evaluate_ensemble(target_name, target_col):
    print(f"\n{'='*60}")
    print(f"EVALUATING {target_name} - Training/Validation")
    print('='*60)
    
    # Load data
    train_dataset = TrainValDataset(pd.read_csv(TRAIN_CSV, sep=';'), IMG_FOLDER, 
                                   target_col, transform=train_transform, split='train')
    val_dataset = TrainValDataset(pd.read_csv(TRAIN_CSV, sep=';'), IMG_FOLDER, 
                                 target_col, transform=val_transform, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val: {len(val_dataset)}")
    print(f"Train dist: {train_dataset.df['label'].value_counts(normalize=True).round(3)}")
    print(f"Val dist: {val_dataset.df['label'].value_counts(normalize=True).round(3)}")
    
    # Load models
    model_dir = MODEL_DIRS[target_name.lower()]
    models = []
    
    for seed in SEEDS:
        model_path = f"{model_dir}/{target_name.lower()}_silver_seed{seed}_best.pth"
        if os.path.exists(model_path):
            model = SwinEmbryoClassifier().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
    
    if len(models) == 0:
        print(f"❌ No {target_name} models found")
        return None
    
    print(f"✓ Loaded {len(models)}/{len(SEEDS)} models")
    
    # Evaluate on validation set (primary metric)
    all_preds, all_probs, all_labels = evaluate_loader(val_loader, models)
    
    # Full metrics
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = np.nan
    
    cm = confusion_matrix(all_labels, all_preds)
    poor_recall = cm[0,0] / (cm[0].sum() + 1e-10)
    good_recall = cm[1,1] / (cm[1].sum() + 1e-10)
    
    print(f"\n✅ {target_name} VALIDATION RESULTS:")
    print(f"   Accuracy:   {acc*100:.2f}%")
    print(f"   Macro F1:   {f1*100:.2f}%")
    print(f"   ROC-AUC:    {auc:.3f}" if not np.isnan(auc) else "   ROC-AUC: N/A")
    print(f"   Poor Rec:   {poor_recall*100:.2f}%")
    print(f"   Good Rec:   {good_recall*100:.2f}%")
    
    return {
        'target': target_name,
        'val_acc': acc*100,
        'macro_f1': f1*100,
        'roc_auc': auc,
        'poor_recall': poor_recall*100,
        'good_recall': good_recall*100,
        'val_samples': len(val_dataset),
        'cm': cm,
        'models_loaded': len(models)
    }

def evaluate_loader(loader, models):
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            
            batch_outputs = [model(images) for model in models]
            avg_outputs = torch.stack(batch_outputs).mean(dim=0)
            probs = torch.softmax(avg_outputs, dim=1)
            preds = torch.argmax(avg_outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============================================================
# RUN EVALUATION
# ============================================================
results = {}
for target_col, target_name in zip(TARGETS, ['ICM', 'TE', 'EXP']):
    result = evaluate_ensemble(target_name, target_col)
    if result:
        results[target_name] = result

# ============================================================
# SPRINGER TABLE 1 - TRAINING/VALIDATION
# ============================================================
print("\n" + "="*80)
print("SPRINGER TABLE 1: VALIDATION PERFORMANCE (5-MODEL ENSEMBLE)")
print("="*80)

df_results = pd.DataFrame(results).T
table_cols = ['val_acc', 'macro_f1', 'poor_recall', 'good_recall', 'roc_auc']
df_display = df_results[table_cols].round(2)

print(df_display)
df_display.to_csv('/kaggle/working/val_performance_summary.csv')

# ============================================================
# FIGURE 1: CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Validation Set Confusion Matrices - 5-Model Ensemble', fontsize=16, fontweight='bold')

for i, (target, result) in enumerate(results.items()):
    cm = result['cm']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, 
                square=True, linewidths=2)
    axes[i].set_title(f'{target}\nVal Acc={result["val_acc"]:.1f}%', fontweight='bold')
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')

plt.tight_layout()
plt.savefig('/kaggle/working/val_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.savefig('/kaggle/working/val_confusion_matrices.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 2: COMPREHENSIVE METRICS
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

x = np.arange(len(results))
width = 0.35

# Accuracy & F1
accs = [results[t]['val_acc'] for t in results]
f1s = [results[t]['macro_f1'] for t in results]
ax1.bar(x - width/2, accs, width, label='Accuracy', alpha=0.8, color='#FF6B6B')
ax1.bar(x + width/2, f1s, width, label='Macro F1', alpha=0.8, color='#4ECDC4')
ax1.set_ylabel('Score (%)')
ax1.set_title('Validation Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(list(results.keys()))
ax1.legend()

# Recall by class
poor_recalls = [results[t]['poor_recall'] for t in results]
good_recalls = [results[t]['good_recall'] for t in results]
ax2.bar(x - width/2, poor_recalls, width, label='Poor Recall', alpha=0.8, color='#96CEB4')
ax2.bar(x + width/2, good_recalls, width, label='Good Recall', alpha=0.8, color='#FFEAA7')
ax2.axhline(y=95, color='red', linestyle='--', label='Safety Threshold (Poor)')
ax2.set_ylabel('Recall (%)')
ax2.set_title('Per-Class Recall (Clinical Safety)')
ax2.set_xticks(x)
ax2.set_xticklabels(list(results.keys()))
ax2.legend()

plt.tight_layout()
plt.savefig('/kaggle/working/val_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('/kaggle/working/val_metrics_comparison.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# LaTeX TABLE (Springer Ready)
# ============================================================
latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Validation Performance of 5-Model Swin Transformer Ensembles}
\label{tab:val-performance}
\begin{tabular}{lccccc}
\hline
Target & Val Acc & Macro F1 & Poor Recall & Good Recall & ROC-AUC \\
\hline
"""

for target in results:
    r = results[target]
    latex_table += f"{target} & {r['val_acc']:.1f}\% & {r['macro_f1']:.1f}\% & {r['poor_recall']:.1f}\% & {r['good_recall']:.1f}\% & {r['roc_auc']:.3f} \\ \n"

latex_table += r"""
\hline
\end{tabular}
\end{table}
"""

with open('/kaggle/working/springer_table1.tex', 'w') as f:
    f.write(latex_table)

print("\n✅ Springer Files Generated:")
print("   📊 /kaggle/working/val_performance_summary.csv")
print("   🖼️  /kaggle/working/val_confusion_matrices.png")
print("   🖼️  /kaggle/working/val_metrics_comparison.png") 
print("   📝 /kaggle/working/springer_table1.tex")
print("\n🎉 VALIDATION PERFORMANCE DASHBOARD COMPLETE!")
