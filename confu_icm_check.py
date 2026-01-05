"""
ICM EVALUATION - Check Confusion Matrix
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm

# ============================================================
# LOAD VALIDATION DATA
# ============================================================
print("Loading validation data...")

val_dataset = EmbryoDataset(val_df, IMG_FOLDER, val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ============================================================
# LOAD TRAINED MODELS
# ============================================================
print("\nLoading trained ICM models...")

models = []
SEEDS = [42, 123, 456, 789, 2024]

for seed in SEEDS:
    model = SwinEmbryoClassifier().to(device)
    model_path = f"saved_models/uncertainty_ICM/ICM_silver_seed{seed}_best.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    models.append(model)
    print(f"✓ Loaded seed {seed}")

# ============================================================
# GET PREDICTIONS (Simple, No Uncertainty)
# ============================================================
print("\nGetting predictions on validation set...")

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        
        # Average across 5 models
        batch_outputs = []
        for model in models:
            outputs = model(images)
            batch_outputs.append(outputs)
        
        # Average predictions
        avg_outputs = torch.stack(batch_outputs).mean(dim=0)
        _, preds = torch.max(avg_outputs, 1)
        
        all_predictions.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# Convert to numpy
val_preds = np.array(all_predictions)
val_labels = np.array(all_labels)

# ============================================================
# CONFUSION MATRIX & ANALYSIS
# ============================================================
print("\n" + "="*70)
print("ICM EVALUATION RESULTS")
print("="*70)

# Accuracy
accuracy = accuracy_score(val_labels, val_preds)
print(f"\n✓ Overall Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(val_labels, val_preds)
print(f"\nConfusion Matrix:")
print("                 Predicted")
print("                 Poor  Good")
print(f"Actual Poor     {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"       Good     {cm[1,0]:4d}  {cm[1,1]:4d}")

# Per-class metrics
poor_recall = cm[0,0] / (cm[0,0] + cm[0,1])
good_recall = cm[1,1] / (cm[1,0] + cm[1,1])
poor_precision = cm[0,0] / (cm[0,0] + cm[1,0]) if (cm[0,0] + cm[1,0]) > 0 else 0
good_precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0

print(f"\nPer-Class Performance:")
print(f"  Poor (ICM < 2):")
print(f"    Recall (Sensitivity):  {poor_recall*100:.1f}%")
print(f"    Precision:             {poor_precision*100:.1f}%")
print(f"  Good (ICM >= 2):")
print(f"    Recall (Sensitivity):  {good_recall*100:.1f}%")
print(f"    Precision:             {good_precision*100:.1f}%")

# Classification Report
print(f"\n{'Full Classification Report':^70}")
print("-" * 70)
print(classification_report(val_labels, val_preds, 
                          target_names=['Poor (ICM<2)', 'Good (ICM>=2)'],
                          digits=4))

# ============================================================
# SANITY CHECKS
# ============================================================
print("\n" + "="*70)
print("SANITY CHECKS")
print("="*70)

# Check if predicting only one class
unique_preds = np.unique(val_preds)
print(f"\nUnique predictions: {unique_preds}")
print(f"  → Model predicts {len(unique_preds)} different classes")

if len(unique_preds) == 1:
    print("  ⚠️⚠️⚠️ WARNING: Model only predicts ONE class! ⚠️⚠️⚠️")
else:
    print("  ✓ Model predicts both classes")

# Check balance
pred_counts = np.bincount(val_preds)
print(f"\nPrediction distribution:")
print(f"  Predicted Poor: {pred_counts[0]} ({pred_counts[0]/len(val_preds)*100:.1f}%)")
print(f"  Predicted Good: {pred_counts[1]} ({pred_counts[1]/len(val_preds)*100:.1f}%)")

# Compare to baseline
baseline_acc = val_labels.value_counts().max() / len(val_labels) * 100
print(f"\nBaseline (majority class): {max(np.bincount(val_labels))/len(val_labels)*100:.1f}%")
print(f"Your model accuracy: {accuracy*100:.2f}%")
print(f"Improvement over baseline: {(accuracy*100) - (max(np.bincount(val_labels))/len(val_labels)*100):.2f}%")

print("\n" + "="*70)
if good_recall > 0.80 and poor_recall > 0.80:
    print("✅ REAL PERFORMANCE! Both classes learned well!")
elif good_recall < 0.50:
    print("⚠️ WARNING: Poor performance on Good class (minority)")
else:
    print("✓ Performance looks reasonable")
print("="*70)
