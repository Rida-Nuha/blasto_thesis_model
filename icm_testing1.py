"""
ICM TEST SET EVALUATION
Evaluate ICM model on completely unseen test data
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
    accuracy_score, precision_recall_fscore_support, 
    confusion_matrix, classification_report, roc_auc_score
)
from tqdm import tqdm
import os

# ============================================================
# CONFIGURATION
# ============================================================
TEST_CSV = "/kaggle/input/dataset/Gardner_test_gold.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "saved_models/uncertainty_ICM"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"✅ Using device: {device}")

# ============================================================
# DATASET CLASS
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
# MODEL CLASS
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
# LOAD TEST DATA
# ============================================================
print("\n" + "="*70)
print("LOADING TEST DATA")
print("="*70)

test_df = pd.read_csv(TEST_CSV, sep=';')
print(f"✓ Loaded test file: {len(test_df)} total samples")

# Filter valid ICM_gold values
test_df_clean = test_df[test_df['ICM_gold'].notna()].copy()
test_df_clean = test_df_clean[test_df_clean['ICM_gold'] != 'ND'].copy()
test_df_clean = test_df_clean[test_df_clean['ICM_gold'] != 'NA'].copy()

# Convert to numeric
test_df_clean['ICM_gold'] = pd.to_numeric(test_df_clean['ICM_gold'], errors='coerce')
test_df_clean = test_df_clean[test_df_clean['ICM_gold'].notna()].copy()

print(f"\nValid ICM samples: {len(test_df_clean)}")
print(f"\nICM_gold distribution:")
print(test_df_clean['ICM_gold'].value_counts().sort_index())

# Binary conversion (same as training)
test_df_clean['label'] = test_df_clean['ICM_gold'].apply(lambda x: 1 if x >= 2 else 0)

print(f"\nBinary label distribution:")
print(f"  Poor (ICM < 2): {(test_df_clean['label']==0).sum()} ({(test_df_clean['label']==0).sum()/len(test_df_clean)*100:.1f}%)")
print(f"  Good (ICM >= 2): {(test_df_clean['label']==1).sum()} ({(test_df_clean['label']==1).sum()/len(test_df_clean)*100:.1f}%)")

# ============================================================
# TEST TRANSFORMS
# ============================================================
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create test dataset
test_dataset = EmbryoDataset(test_df_clean, IMG_FOLDER, test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ============================================================
# LOAD TRAINED MODELS
# ============================================================
print("\n" + "="*70)
print("LOADING TRAINED ICM MODELS")
print("="*70)

models = []
SEEDS = [42, 123, 456, 789, 2024]

for seed in SEEDS:
    model = SwinEmbryoClassifier().to(device)
    model_path = f"{MODEL_DIR}/ICM_silver_seed{seed}_best.pth"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        models.append(model)
        print(f"✓ Loaded seed {seed}")
    else:
        print(f"⚠️ Model not found: {model_path}")

print(f"\n✓ Loaded {len(models)} models")

# ============================================================
# TEST SET EVALUATION
# ============================================================
print("\n" + "="*70)
print("EVALUATING ON TEST SET")
print("="*70)

all_predictions = []
all_probabilities = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device)
        
        # Ensemble prediction from 5 models
        batch_outputs = []
        for model in models:
            outputs = model(images)
            batch_outputs.append(outputs)
        
        # Average predictions
        avg_outputs = torch.stack(batch_outputs).mean(dim=0)
        probs = torch.softmax(avg_outputs, dim=1)
        _, preds = torch.max(avg_outputs, 1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_probabilities.extend(probs[:, 1].cpu().numpy())  # Probability of "Good" class
        all_labels.extend(labels.numpy())

test_preds = np.array(all_predictions)
test_probs = np.array(all_probabilities)
test_labels = np.array(all_labels)

# ============================================================
# FINAL TEST RESULTS
# ============================================================
print("\n" + "="*70)
print("FINAL TEST SET RESULTS - ICM")
print("="*70)

# Overall accuracy
test_acc = accuracy_score(test_labels, test_preds)
print(f"\n✓ Test Accuracy: {test_acc*100:.2f}%")

# Confusion Matrix
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
print(f"  Poor (ICM < 2):")
print(f"    Recall:    {poor_recall*100:.2f}%")
print(f"    Precision: {poor_precision*100:.2f}%")
print(f"    F1-Score:  {2*poor_precision*poor_recall/(poor_precision+poor_recall)*100:.2f}%")
print(f"  Good (ICM >= 2):")
print(f"    Recall:    {good_recall*100:.2f}%")
print(f"    Precision: {good_precision*100:.2f}%")
print(f"    F1-Score:  {2*good_precision*good_recall/(good_precision+good_recall)*100:.2f}%")

# Classification report
print(f"\n{'Detailed Classification Report':^70}")
print("-" * 70)
print(classification_report(test_labels, test_preds, 
                          target_names=['Poor (ICM<2)', 'Good (ICM>=2)'],
                          digits=4))

# ROC-AUC
if len(np.unique(test_labels)) > 1:
    auc = roc_auc_score(test_labels, test_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

# ============================================================
# COMPARISON WITH PAPER
# ============================================================
print("\n" + "="*70)
print("COMPARISON WITH STATE-OF-THE-ART")
print("="*70)

paper_icm = 63.69
improvement = (test_acc * 100) - paper_icm

print(f"\nPaper (2025) - ICM: {paper_icm:.2f}%")
print(f"Your Model   - ICM: {test_acc*100:.2f}%")
print(f"Improvement:        {improvement:+.2f} percentage points")
print(f"Relative Gain:      {(improvement/paper_icm)*100:+.2f}%")

# ============================================================
# VALIDATION CHECK
# ============================================================
print("\n" + "="*70)
print("VALIDATION CHECK")
print("="*70)

# Check if both classes are predicted
unique_preds = np.unique(test_preds)
print(f"\nUnique predictions: {unique_preds}")
if len(unique_preds) == 2:
    print("✓ Model predicts both classes")
else:
    print("⚠️ Model only predicts one class!")

# Check against baseline
baseline_acc = max(np.bincount(test_labels))/len(test_labels)*100
print(f"\nBaseline (majority class): {baseline_acc:.2f}%")
print(f"Your model:                {test_acc*100:.2f}%")
print(f"Improvement over baseline: {(test_acc*100) - baseline_acc:+.2f}%")

# Overall assessment
print("\n" + "="*70)
if good_recall > 0.80 and poor_recall > 0.80 and test_acc > 0.90:
    print("🎉 EXCEPTIONAL TEST PERFORMANCE!")
    print("   ✓ Both classes learned well (>80% recall)")
    print("   ✓ Overall accuracy > 90%")
    print("   ✓ Results are VALID and PUBLICATION-READY!")
elif good_recall > 0.70 and poor_recall > 0.70:
    print("✅ EXCELLENT TEST PERFORMANCE!")
    print("   ✓ Both classes learned well")
    print("   ✓ Results are valid")
else:
    print("✓ Good test performance")
    if good_recall < 0.50 or poor_recall < 0.50:
        print("   ⚠️ One class has low recall - check for bias")
print("="*70)

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("SAVING TEST RESULTS")
print("="*70)

# Save predictions
results_df = test_df_clean.copy()
results_df['predicted_label'] = test_preds
results_df['probability_good'] = test_probs
results_df['correct'] = (test_preds == test_labels).astype(int)

output_file = 'ICM_test_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\n✓ Test predictions saved to: {output_file}")

# Save metrics summary
with open('ICM_test_metrics.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("ICM TEST SET RESULTS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Test Accuracy: {test_acc*100:.2f}%\n\n")
    f.write("Confusion Matrix:\n")
    f.write(f"  Poor predicted: {cm[0,0]} correct, {cm[0,1]} wrong\n")
    f.write(f"  Good predicted: {cm[1,1]} correct, {cm[1,0]} wrong\n\n")
    f.write(f"Poor Recall: {poor_recall*100:.2f}%\n")
    f.write(f"Good Recall: {good_recall*100:.2f}%\n")
    f.write(f"ROC-AUC: {auc:.4f}\n\n")
    f.write(f"Paper ICM: {paper_icm:.2f}%\n")
    f.write(f"Your ICM:  {test_acc*100:.2f}%\n")
    f.write(f"Improvement: {improvement:+.2f}%\n")

print(f"✓ Metrics summary saved to: ICM_test_metrics.txt")

print("\n✅ TEST EVALUATION COMPLETE!")
