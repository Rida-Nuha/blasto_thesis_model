"""
Comprehensive Uncertainty Evaluation with Full Metrics
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC Curve
- Confusion Matrix
- Uncertainty Analysis
- Calibration Analysis
- Clinical Decision Support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


# ============================================================
# CONFIG
# ============================================================
TARGET_SCORE = "EXP_silver"  # Change to match training

TRAIN_CSV = "/kaggle/input/dataset/Gardner_train_silver.csv"
TEST_CSV = "/kaggle/input/dataset/Gardner_test_silver.csv"
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"
MODEL_DIR = "saved_models/uncertainty"
OUTPUT_DIR = "uncertainty_results"

BINARY_THRESHOLD = 2
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_WORKERS = 2
MC_SAMPLES = 20  # MC Dropout iterations

SEEDS = [42, 123, 456, 789, 2024]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATASET
# ============================================================
class GardnerDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_folder, target_column, threshold=2, transform=None):
        self.df = pd.read_csv(csv_file, sep=';')
        self.img_folder = img_folder
        self.target_column = target_column
        self.transform = transform
        
        self.df['binary_label'] = (self.df[target_column] >= threshold).astype(int)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Image'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, row['binary_label']


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ============================================================
# MODEL
# ============================================================
class SwinWithUncertainty(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = self.backbone.head.in_features
        
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def mc_predict(self, x, n_samples=20):
        self.eval()
        self.enable_dropout()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        entropy = -torch.sum(mean_pred * torch.log(mean_pred + 1e-10), dim=1)
        variance = predictions.var(dim=0).mean(dim=1)
        
        return mean_pred, entropy, variance


# ============================================================
# ENSEMBLE PREDICTION WITH UNCERTAINTY
# ============================================================
def ensemble_predict_with_uncertainty(models, loader, device, mc_samples=20):
    """Comprehensive prediction with all metrics"""
    
    all_ensemble_probs = []
    all_mc_entropies = []
    all_mc_variances = []
    all_labels = []
    
    for images, labels in tqdm(loader, desc="Ensemble + MC Dropout prediction"):
        images = images.to(device)
        
        batch_ensemble_probs = []
        batch_mc_entropies = []
        batch_mc_variances = []
        
        for model in models:
            mean_pred, entropy, variance = model.mc_predict(images, n_samples=mc_samples)
            
            batch_ensemble_probs.append(mean_pred.cpu())
            batch_mc_entropies.append(entropy.cpu())
            batch_mc_variances.append(variance.cpu())
        
        # Average across ensemble
        ensemble_probs = torch.stack(batch_ensemble_probs).mean(dim=0)
        mc_entropy = torch.stack(batch_mc_entropies).mean(dim=0)
        mc_variance = torch.stack(batch_mc_variances).mean(dim=0)
        
        all_ensemble_probs.append(ensemble_probs)
        all_mc_entropies.append(mc_entropy)
        all_mc_variances.append(mc_variance)
        all_labels.extend(labels.numpy())
    
    all_ensemble_probs = torch.cat(all_ensemble_probs)
    all_mc_entropies = torch.cat(all_mc_entropies)
    all_mc_variances = torch.cat(all_mc_variances)
    all_labels = np.array(all_labels)
    
    # Final predictions
    predictions = torch.argmax(all_ensemble_probs, dim=1).numpy()
    confidences = torch.max(all_ensemble_probs, dim=1)[0].numpy()
    probs_class1 = all_ensemble_probs[:, 1].numpy()  # For AUC
    
    total_uncertainty = all_mc_entropies.numpy()
    
    return predictions, all_labels, total_uncertainty, confidences, all_ensemble_probs.numpy(), probs_class1


# ============================================================
# COMPREHENSIVE METRICS
# ============================================================
def compute_all_metrics(predictions, labels, probs_class1):
    """Compute all classification metrics"""
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(labels, predictions)
    metrics['precision'] = precision_score(labels, predictions, average='binary')
    metrics['recall'] = recall_score(labels, predictions, average='binary')
    metrics['f1_score'] = f1_score(labels, predictions, average='binary')
    
    # AUC-ROC
    metrics['auc_roc'] = roc_auc_score(labels, probs_class1)
    
    # Per-class metrics
    metrics['precision_per_class'] = precision_score(labels, predictions, average=None)
    metrics['recall_per_class'] = recall_score(labels, predictions, average=None)
    metrics['f1_per_class'] = f1_score(labels, predictions, average=None)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(labels, predictions)
    
    return metrics


# ============================================================
# VISUALIZATION
# ============================================================
def plot_comprehensive_results(predictions, labels, uncertainties, confidences, probs_class1, output_dir):
    """Generate all visualization plots"""
    
    os.makedirs(output_dir, exist_ok=True)
    correct = (predictions == labels)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    
    # ========================================
    # PLOT 1: Confusion Matrix
    # ========================================
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Poor (0)', 'Good (1)'],
                yticklabels=['Poor (0)', 'Good (1)'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {TARGET_SCORE}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add accuracy per class
    for i in range(2):
        acc = cm[i, i] / cm[i].sum() * 100
        plt.text(i+0.5, i-0.3, f'{acc:.1f}%', ha='center', va='center', 
                color='red', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # PLOT 2: ROC Curve
    # ========================================
    fpr, tpr, thresholds = roc_curve(labels, probs_class1)
    auc = roc_auc_score(labels, probs_class1)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {TARGET_SCORE}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # PLOT 3: Uncertainty Distribution
    # ========================================
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(uncertainties[correct], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
    plt.hist(uncertainties[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    plt.xlabel('Uncertainty (Entropy)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.hist(confidences[correct], bins=50, alpha=0.7, label='Correct', color='green', edgecolor='black')
    plt.hist(confidences[~correct], bins=50, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.title('Confidence Distribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.scatter(confidences[correct], uncertainties[correct], alpha=0.5, s=20, 
               c='green', label='Correct', edgecolors='none')
    plt.scatter(confidences[~correct], uncertainties[~correct], alpha=0.8, s=30, 
               c='red', label='Incorrect', edgecolors='black', linewidths=0.5)
    plt.xlabel('Confidence', fontsize=11)
    plt.ylabel('Uncertainty', fontsize=11)
    plt.title('Confidence vs Uncertainty', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # PLOT 4: Rejection Curve
    # ========================================
    rejection_rates = np.linspace(0, 0.5, 100)
    accuracies = []
    baseline_acc = 100 * correct.sum() / len(correct)
    
    for rate in rejection_rates:
        if rate == 0:
            accuracies.append(baseline_acc)
        else:
            threshold = np.percentile(uncertainties, (1 - rate) * 100)
            retained = uncertainties <= threshold
            if retained.sum() > 0:
                acc = 100 * correct[retained].sum() / retained.sum()
                accuracies.append(acc)
            else:
                accuracies.append(accuracies[-1] if accuracies else baseline_acc)
    
    plt.figure(figsize=(10, 6))
    plt.plot(rejection_rates * 100, accuracies, linewidth=3, color='blue', label='Ensemble + MC Dropout')
    plt.axhline(y=baseline_acc, color='red', linestyle='--', linewidth=2, label=f'Baseline: {baseline_acc:.2f}%')
    
    # Highlight key points
    for reject_pct in [10, 20]:
        idx = int(reject_pct)
        plt.plot(reject_pct, accuracies[idx], 'o', markersize=10, color='orange')
        improvement = accuracies[idx] - baseline_acc
        plt.annotate(f'{accuracies[idx]:.2f}% (+{improvement:.2f}%)', 
                    xy=(reject_pct, accuracies[idx]), 
                    xytext=(reject_pct+5, accuracies[idx]-1),
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    plt.xlabel('Rejection Rate (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy vs Rejection Rate - {TARGET_SCORE}', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 50])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rejection_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # PLOT 5: Calibration Curve
    # ========================================
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            bin_accuracies.append(bin_accuracy)
            bin_confidences.append(bin_confidence)
            bin_counts.append(in_bin.sum())
    
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    plt.plot(bin_confidences, bin_accuracies, 'o-', linewidth=2, markersize=10, 
            color='blue', label='Model Calibration')
    
    # Add bin counts as text
    for i, (conf, acc, count) in enumerate(zip(bin_confidences, bin_accuracies, bin_counts)):
        plt.text(conf, acc-0.05, f'n={count}', ha='center', fontsize=8, color='gray')
    
    # Expected Calibration Error
    ece = np.average(np.abs(np.array(bin_accuracies) - np.array(bin_confidences)), 
                    weights=bin_counts)
    
    plt.xlabel('Confidence', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'Calibration Plot - {TARGET_SCORE}\nECE = {ece:.4f}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'calibration_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # PLOT 6: Per-Class Performance
    # ========================================
    cm = confusion_matrix(labels, predictions)
    class_names = ['Poor (0)', 'Good (1)']
    
    # Calculate per-class metrics
    precision_per_class = precision_score(labels, predictions, average=None)
    recall_per_class = recall_score(labels, predictions, average=None)
    f1_per_class = f1_score(labels, predictions, average=None)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precision_per_class, width, label='Precision', color='skyblue', edgecolor='black')
    plt.bar(x, recall_per_class, width, label='Recall', color='lightcoral', edgecolor='black')
    plt.bar(x + width, f1_per_class, width, label='F1-Score', color='lightgreen', edgecolor='black')
    
    # Add value labels on bars
    for i in range(len(class_names)):
        plt.text(i - width, precision_per_class[i] + 0.02, f'{precision_per_class[i]:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
        plt.text(i, recall_per_class[i] + 0.02, f'{recall_per_class[i]:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
        plt.text(i + width, f1_per_class[i] + 0.02, f'{f1_per_class[i]:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(f'Per-Class Performance - {TARGET_SCORE}', fontsize=14, fontweight='bold')
    plt.xticks(x, class_names)
    plt.legend(fontsize=11)
    plt.ylim([0, 1.1])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All plots saved to: {output_dir}")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print(f"COMPREHENSIVE UNCERTAINTY EVALUATION - {TARGET_SCORE}")
    print("="*70 + "\n")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load test dataset
    print(f"Loading test dataset: {TEST_CSV}")
    test_dataset = GardnerDataset(TEST_CSV, IMG_FOLDER, TARGET_SCORE, 
                                  threshold=BINARY_THRESHOLD, transform=val_transform)
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load ensemble models
    print(f"\nLoading {len(SEEDS)} models from {MODEL_DIR}...")
    models = []
    
    for seed in SEEDS:
        model = SwinWithUncertainty(NUM_CLASSES, 0.3)
        model_path = os.path.join(MODEL_DIR, f"{TARGET_SCORE}_seed{seed}_best.pth")
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue
        
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.to(DEVICE)
        model.eval()
        models.append(model)
        print(f"✓ Loaded model with seed {seed}")
    
    if len(models) == 0:
        print("\n❌ No models found! Run train_with_uncertainty.py first.")
        return
    
    print(f"\n✅ {len(models)} models loaded successfully")
    
    # Ensemble + MC Dropout prediction
    print("\n" + "="*70)
    print("RUNNING ENSEMBLE + MC DROPOUT PREDICTION")
    print("="*70 + "\n")
    
    predictions, labels, uncertainties, confidences, probs, probs_class1 = ensemble_predict_with_uncertainty(
        models, test_loader, DEVICE, MC_SAMPLES
    )
    
    # Compute all metrics
    print("\n" + "="*70)
    print("COMPREHENSIVE METRICS")
    print("="*70 + "\n")
    
    metrics = compute_all_metrics(predictions, labels, probs_class1)
    
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    print(f"{'Accuracy':<25} {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"{'Precision':<25} {metrics['precision']:.4f}")
    print(f"{'Recall':<25} {metrics['recall']:.4f}")
    print(f"{'F1-Score':<25} {metrics['f1_score']:.4f}")
    print(f"{'AUC-ROC':<25} {metrics['auc_roc']:.4f}")
    
    print(f"\n{'Per-Class Metrics':^40}")
    print("-" * 40)
    for i, class_name in enumerate(['Poor (0)', 'Good (1)']):
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"  Recall:    {metrics['recall_per_class'][i]:.4f}")
        print(f"  F1-Score:  {metrics['f1_per_class'][i]:.4f}")
    
    print(f"\n{'Confusion Matrix':^40}")
    print("-" * 40)
    cm = metrics['confusion_matrix']
    print(f"                 Predicted")
    print(f"                 Poor  Good")
    print(f"Actual Poor     {cm[0,0]:5d} {cm[0,1]:5d}")
    print(f"       Good     {cm[1,0]:5d} {cm[1,1]:5d}")
    
    # Uncertainty analysis
    correct = (predictions == labels)
    
    print(f"\n{'Uncertainty Analysis':^40}")
    print("-" * 40)
    print(f"Correct predictions:")
    print(f"  Mean uncertainty: {uncertainties[correct].mean():.4f}")
    print(f"  Mean confidence:  {confidences[correct].mean():.4f}")
    print(f"\nIncorrect predictions:")
    print(f"  Mean uncertainty: {uncertainties[~correct].mean():.4f}")
    print(f"  Mean confidence:  {confidences[~correct].mean():.4f}")
    print(f"\nUncertainty ratio (Incorrect/Correct): {uncertainties[~correct].mean()/uncertainties[correct].mean():.2f}x")
    
    # Rejection analysis
    print(f"\n{'Rejection-Based Accuracy':^40}")
    print("-" * 40)
    for reject_pct in [5, 10, 15, 20]:
        threshold = np.percentile(uncertainties, (1 - reject_pct/100) * 100)
        retained = uncertainties <= threshold
        if retained.sum() > 0:
            acc = 100 * correct[retained].sum() / retained.sum()
            improvement = acc - metrics['accuracy']*100
            print(f"Reject {reject_pct:2d}%: {acc:.2f}% (Δ +{improvement:.2f}%)")
    
    # Save detailed results
    results_df = pd.DataFrame({
        'prediction': predictions,
        'true_label': labels,
        'correct': correct,
        'uncertainty': uncertainties,
        'confidence': confidences,
        'prob_poor': probs[:, 0],
        'prob_good': probs[:, 1]
    })
    
    output_csv = os.path.join(OUTPUT_DIR, f'{TARGET_SCORE}_detailed_results.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed results saved to: {output_csv}")
    
    # Save summary metrics
    summary = {
        'target': TARGET_SCORE,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1_score': metrics['f1_score'],
        'auc_roc': metrics['auc_roc'],
        'mean_uncertainty_correct': uncertainties[correct].mean(),
        'mean_uncertainty_incorrect': uncertainties[~correct].mean(),
        'uncertainty_ratio': uncertainties[~correct].mean()/uncertainties[correct].mean()
    }
    
    summary_df = pd.DataFrame([summary])
    summary_csv = os.path.join(OUTPUT_DIR, f'{TARGET_SCORE}_summary_metrics.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"✅ Summary metrics saved to: {summary_csv}")
    
    # Generate all plots
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    plot_comprehensive_results(predictions, labels, uncertainties, confidences, probs_class1, OUTPUT_DIR)
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70 + "\n")
    
    print(f"✅ Test Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"✅ AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"✅ F1-Score: {metrics['f1_score']:.4f}")
    
    if metrics['accuracy'] >= 0.95:
        print(f"\n🌟 OUTSTANDING! Thesis-quality results!")
    elif metrics['accuracy'] >= 0.90:
        print(f"\n✨ EXCELLENT! Publication-worthy results!")
    elif metrics['accuracy'] >= 0.85:
        print(f"\n✓ Very good results!")
    
    print(f"\n📊 Generated plots:")
    print(f"   - Confusion Matrix")
    print(f"   - ROC Curve")
    print(f"   - Uncertainty Analysis")
    print(f"   - Rejection Curve")
    print(f"   - Calibration Plot")
    print(f"   - Per-Class Performance")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
