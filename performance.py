"""
Training/Validation Performance Metrics - ICM/TE/EXP Ensembles
COMPLETE VERSION - ALL SYNTAX ERRORS FIXED
Springer Table 1 + Figures - Training/Validation Only
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
# CONFIGURATION - FIXED KEYS
# ============================================================
TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"

MODEL_DIRS = {
    'icm': "/kaggle/working/saved_models/uncertainty",
    'te':  "/kaggle/working/saved_models/uncertainty", 
    'exp': "/kaggle/working/saved_models/uncertainty"
}

TARGETS = ['ICM_silver', 'TE_silver', 'EXP_silver']
SEEDS = [42, 123, 456, 789, 2024]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

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
# DATASET CLASS
# ============================================================
class TrainValDataset(Dataset):
    def __init__(self, csv_file, img_folder, target_col, threshold=2, transform=None, split='train', seed=42):
        self.df = pd.read_csv(csv_file, sep=';')
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
        
        # Stratified split (85/15)
        if split == 'train':
            train_df = self.df.groupby('label', group_keys=False).apply(
                lambda x: x.sample(frac=0.85, random_state=seed)
            ).reset_index(drop=True)
            self.df = train_df
        else:  # validation
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
def evaluate_loader(loader, models, target_name):
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Eval {target_name}"):
            images = images.to(device)
            
            # Ensemble prediction
            batch_outputs = []
            for model in models:
                outputs = model(images)
                batch_outputs.append(outputs)
            
            avg_outputs = torch.stack(batch_outputs).mean(dim=0)
            probs = torch.softmax(avg_outputs, dim=1)
            preds = torch.argmax(avg_outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)

def compute_metrics(preds, probs, labels):
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, probs)
    else:
        auc = np.nan
    
    cm = confusion_matrix(labels, preds)
    poor_recall = cm[0,0] / (cm[0].sum() + 1e-10)
    good_recall = cm[1,1] / (cm[1].sum() + 1e-10)
    
    return {
        'accuracy': acc,
        'macro_precision': prec,
        'macro_recall': rec,
        'macro_f1': f1,
        'roc_auc': auc,
        'poor_recall': poor_recall,
        'good_recall': good_recall,
        'confusion_matrix': cm
    }

# ============================================================
# MAIN EVALUATION FUNCTION - FIXED RETURN
# ============================================================
def evaluate_ensemble(target_key, target_col):
    print(f"\n{'='*60}")
    print(f"EVALUATING {target_key.upper()} - Training/Validation")
    print('='*60)
    
    # Create datasets
    val_dataset = TrainValDataset(TRAIN_CSV, IMG_FOLDER, target_col, transform=val_transform, split='val')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Val distribution:\n{val_dataset.df['label'].value_counts(normalize=True).round(3)}")
    
    # Load models
    model_dir = MODEL_DIRS[target_key]
    models = []
    
    for seed in SEEDS:
        model_path = f"{model_dir}/{target_key}_silver_seed{seed}_best.pth"
        if os.path.exists(model_path):
            model = SwinEmbryoClassifier().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            models.append(model)
            print(f"✓ Loaded seed {seed}")
        else:
            print(f"⚠️ Missing: {model_path}")
    
    if len(models) == 0:
        print(f"❌ No models found for {target_key}")
        return None
    
    print(f"✓ Loaded {len(models)}/{len(SEEDS)} models")
    
    # Evaluate validation set
    preds, probs, labels = evaluate_loader(val_loader, models, target_key)
    metrics = compute_metrics(preds, probs, labels)
    
    print(f"\n✅ VALIDATION RESULTS ({len(labels)} samples):")
    print(f"   Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f"   Macro F1:      {metrics['macro_f1']*100:.2f}%")
    print(f"   ROC-AUC:       {metrics['roc_auc']:.3f}")
    print(f"   Poor Recall:   {metrics['poor_recall']*100:.2f}%")
    print(f"   Good Recall:   {metrics['good_recall']*100:.2f}%")
    
    # FIXED RETURN - no f-string issues
    result = {
        'target': target_key.upper(),
        'accuracy': metrics['accuracy'] * 100,
        'macro_precision': metrics['macro_precision'] * 100,
        'macro_recall': metrics['macro_recall'] * 100,
        'macro_f1': metrics['macro_f1'] * 100,
        'roc_auc': metrics['roc_auc'],
        'poor_recall': metrics['poor_recall'] * 100,
        'good_recall': metrics['good_recall'] * 100,
        'confusion_matrix': metrics['confusion_matrix']
    }
    return result

# ============================================================
# RUN EVALUATION
# ============================================================
print("Starting ensemble evaluation...")
results = {}

for target_col, target_key in zip(TARGETS, ['icm', 'te', 'exp']):
    result = evaluate_ensemble(target_key, target_col)
    if result:
        results[result['target']] = result

# ============================================================
# SPRINGER TABLE 1
# ============================================================
print("\n" + "="*80)
print("SPRINGER TABLE 1: 5-MODEL ENSEMBLE VALIDATION PERFORMANCE")
print("="*80)

df_results = pd.DataFrame(results).T
display_cols = ['accuracy', 'macro_f1', 'poor_recall', 'good_recall', 'roc_auc']
df_display = df_results[display_cols].round(2)

print(df_display)
df_display.to_csv('/kaggle/working/val_performance_table.csv')

# ============================================================
# FIGURE 1: CONFUSION MATRICES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Figure 1: Validation Confusion Matrices (5-Model Ensemble)', fontsize=16, fontweight='bold')

for i, target in enumerate(results.keys()):
    cm = results[target]['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], 
                cbar=False, square=True, linewidths=2)
    axes[i].set_title(f'{target}\nVal Acc={results[target]["accuracy"]:.1f}%', 
                     fontweight='bold', fontsize=12)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('/kaggle/working/val_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.savefig('/kaggle/working/val_confusion_matrices.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# FIGURE 2: PERFORMANCE COMPARISON
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

targets = list(results.keys())
x = np.arange(len(targets))

# Accuracy & F1
ax1.bar(x-0.2, [results[t]['accuracy'] for t in targets], 0.4, label='Accuracy', alpha=0.8, color='#FF6B6B')
ax1.bar(x+0.2, [results[t]['macro_f1'] for t in targets], 0.4, label='Macro F1', alpha=0.8, color='#4ECDC4')
ax1.set_ylabel('Score (%)')
ax1.set_title('Validation Performance')
ax1.set_xticks(x)
ax1.set_xticklabels(targets)
ax1.legend()
ax1.set_ylim(0, 105)

# Recall comparison
ax2.bar(x-0.2, [results[t]['poor_recall'] for t in targets], 0.4, label='Poor Recall', alpha=0.8, color='#96CEB4')
ax2.bar(x+0.2, [results[t]['good_recall'] for t in targets], 0.4, label='Good Recall', alpha=0.8, color='#FFEAA7')
ax2.axhline(y=95, color='red', linestyle='--', label='Safety Threshold')
ax2.set_ylabel('Recall (%)')
ax2.set_title('Per-Class Recall')
ax2.set_xticks(x)
ax2.set_xticklabels(targets)
ax2.legend()
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('/kaggle/working/val_performance_charts.png', dpi=300, bbox_inches='tight')
plt.savefig('/kaggle/working/val_performance_charts.pdf', bbox_inches='tight')
plt.show()

# ============================================================
# FIXED SPRINGER LATE X TABLE - NO F-STRING ISSUES
# ============================================================
latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Validation Performance of 5-Model Swin Transformer Ensembles}
\label{tab:ensemble-performance}
\begin{tabular}{lccccc}
\hline
Target & Accuracy & Macro F1 & Poor Recall & Good Recall & ROC-AUC \\
\hline
"""

for target in results:
    r = results[target]
    latex_table += f"{target} & {r['accuracy']:.1f}\% & {r['macro_f1']:.1f}\% & {r['poor_recall']:.1f}\% & {r['good_recall']:.1f}\% & {r['roc_auc']:.3f} \\\\ \n"

latex_table += r"""
\hline
\end{tabular}
\end{table}
"""

with open('/kaggle/working/springer_table1.tex', 'w') as f:
    f.write(latex_table)

print("\n" + "="*80)
print("✅ ALL FILES GENERATED FOR SPRINGER PAPER:")
print("📊 Table:           /kaggle/working/val_performance_table.csv")
print("🖼️  Confusion Mats: /kaggle/working/val_confusion_matrices.png")
print("🖼️  Charts:         /kaggle/working/val_performance_charts.png")
print("📝 LaTeX Table:     /kaggle/working/springer_table1.tex")
print("🎉 PERFORMANCE EVALUATION COMPLETE!")
