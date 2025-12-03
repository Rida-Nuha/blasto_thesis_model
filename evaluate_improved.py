"""
Evaluation Script for train_improved.py Model

Tests model on Gardner test set and computes:
- Test accuracy
- Per-class accuracy  
- Confusion matrix
- Classification report
- Uncertainty analysis
- Predictions CSV
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import swin_t, Swin_T_Weights
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import os
from tqdm import tqdm

from dataset import BlastocystDataset, val_transform
from test_dataset import BlastocystTestDataset, test_transform


# ============================================================
# MODEL (Must match train_improved.py)
# ============================================================
class SwinUncertaintyImproved(nn.Module):
    """Exact same model from train_improved.py"""
    
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        
        self.backbone = swin_t(weights=None)  # Will load trained weights
        in_features = self.backbone.head.in_features
        
        # Same head as training
        self.backbone.head = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, num_classes)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict_with_uncertainty(self, x, n_samples=10):
        """MC Dropout for uncertainty"""
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        predictions = torch.stack(predictions)
        mean_probs = predictions.mean(dim=0)
        
        # Uncertainty metrics
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        variance = predictions.var(dim=0).mean(dim=1)
        
        self.eval()
        return {
            'probs': mean_probs,
            'uncertainty': entropy,
            'variance': variance,
            'all_predictions': predictions
        }


# ============================================================
# CONFIG
# ============================================================
TEST_GOLD = "data/Gardner_test_gold.xlsx"  # Your test dataset
IMG_FOLDER = "data/images"
MODEL_PATH = "saved_models/swin_improved_best.pth"
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 3
DROPOUT_RATE = 0.3
BATCH_SIZE = 32
MC_SAMPLES = 20  # More samples for better uncertainty


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, test_loader, device, use_uncertainty=True):
    """
    Comprehensive evaluation with uncertainty
    
    Returns:
        results: Dict with all metrics
        predictions_df: DataFrame with predictions
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainties = []
    all_image_names = []
    all_agreements = []
    
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70 + "\n")
    
    with torch.no_grad():
        for images, labels, info in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            
            if use_uncertainty:
                # MC Dropout prediction
                result = model.predict_with_uncertainty(images, n_samples=MC_SAMPLES)
                probs = result['probs']
                uncertainty = result['uncertainty']
            else:
                # Standard prediction
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                uncertainty = torch.zeros(len(labels))
            
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_uncertainties.extend(uncertainty.cpu().numpy())
            all_image_names.extend(info['image_name'])
            all_agreements.extend([a.item() if a is not None else None for a in info['agreement']])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_uncertainties = np.array(all_uncertainties)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Per-class accuracy
    per_class_acc = []
    class_names = ['Poor (0)', 'Medium (1)', 'Good (3)']
    
    for i in range(NUM_CLASSES):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = (all_preds[mask] == all_labels[mask]).mean()
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    results = {
        'accuracy': accuracy * 100,
        'f1_score': f1,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'uncertainties': all_uncertainties,
        'image_names': all_image_names,
        'agreements': all_agreements
    }
    
    return results


def print_results(results):
    """Pretty print evaluation results"""
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:  {results['accuracy']:.2f}%")
    print(f"   F1 Score:  {results['f1_score']:.4f}")
    
    print(f"\n📈 Per-Class Accuracy:")
    class_names = ['Poor (0)', 'Medium (1)', 'Good (3)']
    for i, (name, acc) in enumerate(zip(class_names, results['per_class_accuracy'])):
        print(f"   {name}: {acc*100:.2f}%")
    
    print(f"\n📋 Classification Report:")
    report = results['classification_report']
    for cls in ['Poor (0)', 'Medium (1)', 'Good (3)']:
        if cls in report:
            print(f"   {cls}:")
            print(f"      Precision: {report[cls]['precision']:.3f}")
            print(f"      Recall:    {report[cls]['recall']:.3f}")
            print(f"      F1-score:  {report[cls]['f1-score']:.3f}")
    
    print(f"\n🎯 Uncertainty Statistics:")
    print(f"   Mean uncertainty: {results['uncertainties'].mean():.4f}")
    print(f"   Std uncertainty:  {results['uncertainties'].std():.4f}")
    
    # Uncertainty vs Correctness
    correct_mask = results['predictions'] == results['labels']
    correct_uncertainty = results['uncertainties'][correct_mask].mean()
    incorrect_uncertainty = results['uncertainties'][~correct_mask].mean()
    
    print(f"   Uncertainty when correct:   {correct_uncertainty:.4f}")
    print(f"   Uncertainty when incorrect: {incorrect_uncertainty:.4f}")
    
    if incorrect_uncertainty > correct_uncertainty:
        print(f"   ✅ Model is MORE uncertain on errors (good!)")
    else:
        print(f"   ⚠️  Model confidence not well-calibrated")
    
    print("="*70 + "\n")


def plot_confusion_matrix(cm, save_path='results/figures/confusion_matrix.png'):
    """Plot and save confusion matrix"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2%',
        cmap='Blues',
        xticklabels=['Poor (0)', 'Medium (1)', 'Good (3)'],
        yticklabels=['Poor (0)', 'Medium (1)', 'Good (3)'],
        cbar_kws={'label': 'Proportion'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved: {save_path}")
    plt.close()


def plot_uncertainty_analysis(results, save_path='results/figures/uncertainty_analysis.png'):
    """Plot uncertainty analysis"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    correct_mask = results['predictions'] == results['labels']
    
    # 1. Uncertainty distribution
    ax = axes[0, 0]
    ax.hist(results['uncertainties'][correct_mask], bins=30, alpha=0.7, 
            label='Correct', color='green', edgecolor='black')
    ax.hist(results['uncertainties'][~correct_mask], bins=30, alpha=0.7, 
            label='Incorrect', color='red', edgecolor='black')
    ax.set_xlabel('Uncertainty (Entropy)')
    ax.set_ylabel('Frequency')
    ax.set_title('Uncertainty Distribution: Correct vs Incorrect')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. Uncertainty by class
    ax = axes[0, 1]
    class_names = ['Poor (0)', 'Medium (1)', 'Good (3)']
    uncertainties_by_class = [
        results['uncertainties'][results['labels'] == i] 
        for i in range(NUM_CLASSES)
    ]
    ax.boxplot(uncertainties_by_class, labels=class_names)
    ax.set_ylabel('Uncertainty')
    ax.set_title('Uncertainty by True Class')
    ax.grid(alpha=0.3)
    
    # 3. Confidence vs Accuracy
    ax = axes[1, 0]
    max_probs = results['probabilities'].max(axis=1)
    
    # Bin by confidence
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accuracies = []
    
    for i in range(len(bins)-1):
        mask = (max_probs >= bins[i]) & (max_probs < bins[i+1])
        if mask.sum() > 0:
            acc = (results['predictions'][mask] == results['labels'][mask]).mean()
            bin_accuracies.append(acc)
        else:
            bin_accuracies.append(0)
    
    ax.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Per-class confidence
    ax = axes[1, 1]
    for i, name in enumerate(class_names):
        mask = results['predictions'] == i
        if mask.sum() > 0:
            confidences = results['probabilities'][mask, i]
            ax.hist(confidences, bins=20, alpha=0.5, label=name, edgecolor='black')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Uncertainty analysis saved: {save_path}")
    plt.close()


def save_predictions(results, save_path='results/predictions.csv'):
    """Save predictions to CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    predictions_df = pd.DataFrame({
        'image': results['image_names'],
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'correct': results['predictions'] == results['labels'],
        'prob_class_0': results['probabilities'][:, 0],
        'prob_class_1': results['probabilities'][:, 1],
        'prob_class_2': results['probabilities'][:, 2],
        'max_probability': results['probabilities'].max(axis=1),
        'uncertainty': results['uncertainties'],
        'agreement': results['agreements']
    })
    
    predictions_df.to_csv(save_path, index=False)
    print(f"✅ Predictions saved: {save_path}")
    
    return predictions_df


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("MODEL EVALUATION - SWIN TRANSFORMER IMPROVED")
    print("="*70 + "\n")
    
    print(f"Device: {DEVICE}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = BlastocystTestDataset(
        gold_xlsx=TEST_GOLD,
        root_dir=IMG_FOLDER,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    model = SwinUncertaintyImproved(
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE
    )
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("   Make sure you've trained the model first!")
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(DEVICE)
    print("✅ Model loaded successfully")
    
    # Evaluate
    results = evaluate_model(model, test_loader, DEVICE, use_uncertainty=True)
    
    # Print results
    print_results(results)
    
    # Save visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(results['confusion_matrix'])
    plot_uncertainty_analysis(results)
    
    # Save predictions
    predictions_df = save_predictions(results)
    
    # Summary for thesis
    print("\n" + "="*70)
    print("THESIS SUMMARY")
    print("="*70)
    print(f"\n✅ Test Accuracy: {results['accuracy']:.2f}%")
    print(f"✅ Per-class accuracy: {[f'{acc*100:.1f}%' for acc in results['per_class_accuracy']]}")
    print(f"✅ Model shows {'GOOD' if results['uncertainties'][results['predictions'] != results['labels']].mean() > results['uncertainties'][results['predictions'] == results['labels']].mean() else 'POOR'} uncertainty calibration")
    
    if results['accuracy'] >= 80:
        print(f"\n🎉 EXCELLENT! Thesis-quality results!")
    elif results['accuracy'] >= 75:
        print(f"\n✓ Good results for thesis")
    elif results['accuracy'] >= 70:
        print(f"\n✓ Acceptable - focus on uncertainty contribution")
    else:
        print(f"\n⚠️ Consider more training or using full fine-tuning")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
