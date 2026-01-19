"""
EXP TEST SET EVALUATION
Evaluates 5 trained EXP models on Gardner_test_gold.xlsx
Identical to ICM/TE test structure
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import swin_t, Swin_T_Weights
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
from tqdm import tqdm
import os

# ============================================================
# CONFIGURATION - EXP SPECIFIC
# ============================================================
TEST_EXCEL = "/kaggle/input/dataset/Gardner_test_gold.xlsx"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "saved_models/uncertainty_EXP"  # ← CHANGE 1: EXP models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {device}")

# ============================================================
# DATASET CLASS (SAME AS ICM/TE)
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
# MODEL CLASS (SAME AS ICM/TE)
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
# LOAD TEST DATA - EXP SPECIFIC
# ============================================================
print("\n" + "="*70)
print("LOADING EXP TEST DATA")
print("="*70)

test_df = pd.read_excel(TEST_EXCEL)
print(f"✓ Loaded Excel file: {len(test_df)} total samples")

# Filter valid EXP_gold values ← CHANGE 2: EXP_gold
test_df_clean = test_df[test_df['EXP_gold'].notna()].copy()
test_df_clean = test_df_clean[test_df_clean['EXP_gold'] != 'ND'].copy()
test_df_clean = test_df_clean[test_df_clean['EXP_gold'] != 'NA'].copy()

# Convert to numeric
test_df_clean['EXP_gold'] = pd.to_numeric(test_df_clean['EXP_gold'], errors='coerce')
test_df_clean = test_df_clean[test_df_clean['EXP_gold'].notna()].copy()

print(f"\nValid EXP samples: {len(test_df_clean)}")
print(f"\nEXP_gold distribution:")
print(test_df_clean['EXP_gold'].value_counts().sort_index())

# Binary conversion (EXP >= 2 is Good) - SAME LOGIC
test_df_clean['label'] = test_df_clean['EXP_gold'].apply(lambda x: 1 if x >= 2 else 0)

print(f"\nBinary label distribution:")
poor_count = (test_df_clean['label']==0).sum()
good_count = (test_df_clean['label']==1).sum()
print(f"  Poor (EXP < 2): {poor_count} ({poor_count/len(test_df_clean)*100:.1f}%)")
print(f"  Good (EXP >= 2): {good_count} ({good_count/len(test_df_clean)*100:.1f}%)")

# ============================================================
# TEST TRANSFORMS (SAME)
# ============================================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = EmbryoDataset(test_df_clean, IMG_FOLDER, test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ============================================================
# LOAD TRAINED EXP MODELS
# ============================================================
print("\n" + "="*70)
print("LOADING TRAINED EXP MODELS")
print("="*70)

models = []
SEEDS = [42, 123, 456, 789, 2024]

for seed in SEEDS:
    model = SwinEmbryoClassifier().to(device)
    model_path = f"{MODEL_DIR}/EXP_silver_seed{seed}_best.pth"  # ← CHANGE 3
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
        print(f"✓ Loaded seed {seed}")
    else:
        print(f"⚠️ Model not found: {model_path}")

print(f"\n✓ Loaded {len(models)} models")

# ============================================================
# TEST EVALUATION (IDENTICAL)
# ============================================================
print("\n" + "="*70)
print("EVALUATING EXP ON TEST SET")
print("="*70)

all_predictions = []
all_probabilities = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        batch_outputs = []
        for model in models:
            outputs = model(images)
            batch_outputs.append(outputs)
        
        avg_outputs = torch.stack(batch_outputs).mean(dim=0)
        probs = torch.softmax(avg_outputs, dim=1)
        _, preds = torch.max(avg_outputs, 1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(labels.numpy())

test_preds = np.array(all_predictions)
test_probs = np.array(all_probabilities)
test_labels = np.array(all_labels)

# ============================================================
# RESULTS (IDENTICAL)
# ============================================================
print("\n" + "="*70)
print("FINAL TEST SET RESULTS - EXP")
print("="*70)

test_acc = accuracy_score(test_labels, test_preds)
print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")

cm = confusion_matrix(test_labels, test_preds)
print(f"\nConfusion Matrix:")
print("                 Predicted")
print("                 Poor  Good")
print(f"Actual Poor     {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Good     {cm[1,0]:4d}  {cm[1,1]:4d}")

# Per-class metrics
poor_recall = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
good_recall = cm[1,1]/(cm[1,0]+cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0
poor_precision = cm[0,0]/(cm[0,0]+cm[1,0]) if (cm[0,0]+cm[1,0]) > 0 else 0
good_precision = cm[1,1]/(cm[0,1]+cm[1,1]) if (cm[0,1]+cm[1,1]) > 0 else 0

print(f"\nPer-Class Metrics:")
print(f"  Poor (EXP < 2):")
print(f"    Recall:    {poor_recall*100:.2f}%")
print(f"    Precision: {poor_precision*100:.2f}%")
if (poor_precision + poor_recall) > 0:
    poor_f1 = 2*poor_precision*poor_recall/(poor_precision+poor_recall)
    print(f"    F1-Score:  {poor_f1*100:.2f}%")
print(f"  Good (EXP >= 2):")
print(f"    Recall:    {good_recall*100:.2f}%")
print(f"    Precision: {good_precision*100:.2f}%")
if (good_precision + good_recall) > 0:
    good_f1 = 2*good_precision*good_recall/(good_precision+good_recall)
    print(f"    F1-Score:  {good_f1*100:.2f}%")

print(f"\nDetailed Classification Report:")
print(classification_report(test_labels, test_preds, 
                          target_names=['Poor (EXP<2)', 'Good (EXP>=2)'],
                          digits=4))

if len(np.unique(test_labels)) > 1:
    auc = roc_auc_score(test_labels, test_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

# ============================================================
# SOTA COMPARISON
# ============================================================
print("\n" + "="*70)
print("COMPARISON WITH STATE-OF-THE-ART")
print("="*70)

paper_exp = 72.88  # Paper baseline
improvement = (test_acc * 100) - paper_exp

print(f"\nPaper (2025) - EXP: {paper_exp:.2f}%")
print(f"Your Model   - EXP: {test_acc*100:.2f}%")
print(f"Improvement:        {improvement:+.2f} percentage points")
print(f"Relative Gain:      {(improvement/paper_exp)*100:+.2f}%")

# Baseline comparison
baseline_acc = max(np.bincount(test_labels))/len(test_labels)*100
print(f"\nBaseline (majority): {baseline_acc:.2f}%")
print(f"Improvement over baseline: {(test_acc*100) - baseline_acc:+.2f}%")

print("\n" + "="*70)
print("✅ EXP TEST EVALUATION COMPLETE!")
print("🎉 ALL 3 TARGETS DONE!")
print("="*70)

# Save results
results_df = pd.DataFrame({
    'Image': test_df_clean['Image'].values,
    'EXP_gold': test_df_clean['EXP_gold'].values,
    'True_Label': test_labels,
    'Predicted_Label': test_preds,
    'Confidence_Good': test_probs,
    'Correct': test_labels == test_preds
})

results_df.to_csv('EXP_test_predictions.csv', index=False)
print(f"\n✅ Predictions saved: EXP_test_predictions.csv")
