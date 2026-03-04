"""
Swin-T Evaluation Script (CORAL Outputs & MC Dropout)
Final Master's Thesis Evaluation Script
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, precision_recall_fscore_support
import scipy.stats

# ============================================================
# 1. CONFIGURATION
# ============================================================
TEST_DATA_PATH = "/kaggle/input/datasets/ridakhan09/dataset/Gardner_test_gold.xlsx"
IMG_FOLDER = "/kaggle/input/datasets/ridakhan09/dataset/Images/Images"

# 🚨 VERIFY THIS PATH MATCHES YOUR SAVED EPOCH 18 MODEL
MODEL_PATH = "/kaggle/working/saved_models/swin_microscopy_coral/swin_coral_seed42_best.pth"
SAVE_DIR = "/kaggle/working/evaluation_plots_swin_coral"

NUM_CLASSES_EXP = 5  
NUM_CLASSES_ICM = 3  # 3-Class logic enforced (No NA)
NUM_CLASSES_TE = 3   # 3-Class logic enforced (No NA)

DROPOUT_RATE = 0.3  
BATCH_SIZE = 1  
MC_PASSES = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. DATASET (3-Class Enforced)
# ============================================================
class GardnerTestDataset(Dataset):
    def __init__(self, excel_file, img_folder, transform=None):
        print(f"Loading Test Data: {excel_file}...")
        self.df = pd.read_excel(excel_file)
        self.img_folder = img_folder
        self.transform = transform
        
        self.targets = ["EXP_gold", "ICM_gold", "TE_gold"]
        
        for col in self.targets:
            valid_mask = (self.df[col].notna()) & (self.df[col] != 'ND') & (self.df[col] != 'NA')
            self.df = self.df[valid_mask].copy()
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(int)
            
            if col in ["ICM_gold", "TE_gold"]:
                self.df = self.df[self.df[col] < 3].copy() 
                
            self.df = self.df[self.df[col].notna()].copy()
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, str(row['Image']))
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        l_exp = row["EXP_gold"]
        l_icm = row["ICM_gold"]
        l_te = row["TE_gold"]
        
        return image, l_exp, l_icm, l_te

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. SWIN-T ARCHITECTURE
# ============================================================
class MultiTaskMicroscopySwin(nn.Module):
    def __init__(self, weight_path="", dropout_rate=0.3):
        super().__init__()
        self.backbone = swin_t(weights=None) 
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        # Heads structured for CORAL (N-1 outputs)
        self.exp_head = self._make_head(in_features, NUM_CLASSES_EXP - 1, dropout_rate)
        self.icm_head = self._make_head(in_features, NUM_CLASSES_ICM - 1, dropout_rate)
        self.te_head = self._make_head(in_features, NUM_CLASSES_TE - 1, dropout_rate)

    def _make_head(self, in_features, out_features, dropout_rate):
        return nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, out_features)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.exp_head(features), self.icm_head(features), self.te_head(features)

    def enable_dropout(self):
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

# ============================================================
# 4. CORAL TO PROBABILITY & UNCERTAINTY UTILS
# ============================================================
def coral_logits_to_probs(logits):
    """Converts CORAL cumulative logits back into standard class probabilities"""
    cum_probs = torch.sigmoid(logits)
    ones = torch.ones((logits.size(0), 1), device=logits.device)
    zeros = torch.zeros((logits.size(0), 1), device=logits.device)
    padded = torch.cat([ones, cum_probs, zeros], dim=1)
    
    probs = padded[:, :-1] - padded[:, 1:]
    probs = torch.clamp(probs, min=1e-7) 
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs

def calculate_entropy(preds):
    mean_preds = np.mean(preds, axis=0)[0]
    entropy = scipy.stats.entropy(mean_preds)
    return mean_preds, entropy

def predict_with_uncertainty_coral(model, image, passes=MC_PASSES):
    model.eval()
    model.enable_dropout() 
    
    exp_preds, icm_preds, te_preds = [], [], []
    
    with torch.no_grad():
        for _ in range(passes):
            out_exp, out_icm, out_te = model(image)
            
            # Convert logits to probabilities
            exp_probs = coral_logits_to_probs(out_exp)
            icm_probs = coral_logits_to_probs(out_icm)
            te_probs = coral_logits_to_probs(out_te)
            
            exp_preds.append(exp_probs.cpu().numpy())
            icm_preds.append(icm_probs.cpu().numpy())
            te_preds.append(te_probs.cpu().numpy())
            
    exp_preds = np.stack(exp_preds)
    icm_preds = np.stack(icm_preds)
    te_preds = np.stack(te_preds)
    
    exp_mean, exp_ent = calculate_entropy(exp_preds)
    icm_mean, icm_ent = calculate_entropy(icm_preds)
    te_mean, te_ent = calculate_entropy(te_preds)
    
    return (np.argmax(exp_mean), exp_ent), (np.argmax(icm_mean), icm_ent), (np.argmax(te_mean), te_ent)

def plot_confusion_matrix(y_true, y_pred, classes, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, filename), dpi=300)
    plt.close()

# ============================================================
# 5. MAIN EVALUATION LOOP
# ============================================================
def evaluate_mc_dropout():
    print("Loading Model for Evaluation...")
    model = MultiTaskMicroscopySwin(dropout_rate=DROPOUT_RATE).to(DEVICE)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Could not find model at {MODEL_PATH}")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    test_ds = GardnerTestDataset(TEST_DATA_PATH, IMG_FOLDER, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    all_exp_true, all_exp_pred = [], []
    all_icm_true, all_icm_pred = [], []
    all_te_true, all_te_pred = [], []
    
    print(f"Running MC Dropout Inference ({MC_PASSES} passes per image)...")
    for image, l_exp, l_icm, l_te in tqdm(test_loader, desc="Evaluating"):
        image = image.to(DEVICE)
        
        (p_exp, _), (p_icm, _), (p_te, _) = predict_with_uncertainty_coral(model, image)
        
        all_exp_true.append(l_exp.item())
        all_exp_pred.append(p_exp)
        
        all_icm_true.append(l_icm.item())
        all_icm_pred.append(p_icm)
        
        all_te_true.append(l_te.item())
        all_te_pred.append(p_te)

    # ---------------------------------------------------------
    # CALCULATE METRICS
    # ---------------------------------------------------------
    exp_acc = accuracy_score(all_exp_true, all_exp_pred)
    exp_qwk = cohen_kappa_score(all_exp_true, all_exp_pred, weights='quadratic')
    
    icm_acc = accuracy_score(all_icm_true, all_icm_pred)
    icm_qwk = cohen_kappa_score(all_icm_true, all_icm_pred, weights='quadratic')
    
    te_acc = accuracy_score(all_te_true, all_te_pred)
    te_qwk = cohen_kappa_score(all_te_true, all_te_pred, weights='quadratic')

    exp_prec, exp_rec, _, _ = precision_recall_fscore_support(all_exp_true, all_exp_pred, average='macro', zero_division=0)
    icm_prec, icm_rec, _, _ = precision_recall_fscore_support(all_icm_true, all_icm_pred, average='macro', zero_division=0)
    te_prec, te_rec, _, _ = precision_recall_fscore_support(all_te_true, all_te_pred, average='macro', zero_division=0)

    print("\n==================================================")
    print("FINAL CLINICAL METRICS (SWIN-T + CORAL)")
    print("==================================================")
    print("\nEXPANSION RESULTS:")
    print(f"  Accuracy: {exp_acc*100:.2f}% | QWK: {exp_qwk:.4f}")
    
    print("\nICM RESULTS:")
    print(f"  Accuracy: {icm_acc*100:.2f}% | QWK: {icm_qwk:.4f}")
    
    print("\nTE RESULTS:")
    print(f"  Accuracy: {te_acc*100:.2f}% | QWK: {te_qwk:.4f}")

    print("\n==================================================")
    print("MACRO-AVERAGED METRICS")
    print("==================================================")
    print(f"EXP - Precision: {exp_prec*100:.2f}% | Recall: {exp_rec*100:.2f}%")
    print(f"ICM - Precision: {icm_prec*100:.2f}% | Recall: {icm_rec*100:.2f}%")
    print(f"TE  - Precision: {te_prec*100:.2f}% | Recall: {te_rec*100:.2f}%")
    print("==================================================\n")

    plot_confusion_matrix(all_exp_true, all_exp_pred, ['1', '2', '3', '4', '5'], 'Swin-T CORAL - Expansion', 'cm_exp_swin.png')
    plot_confusion_matrix(all_icm_true, all_icm_pred, ['A', 'B', 'C'], 'Swin-T CORAL - ICM', 'cm_icm_swin.png')
    plot_confusion_matrix(all_te_true, all_te_pred, ['A', 'B', 'C'], 'Swin-T CORAL - TE', 'cm_te_swin.png')
    
    print(f"✅ All evaluation plots saved successfully to: {SAVE_DIR}")

if __name__ == "__main__":
    evaluate_mc_dropout()
