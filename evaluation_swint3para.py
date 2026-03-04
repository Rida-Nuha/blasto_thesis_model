"""
Multi-Task Evaluation & Uncertainty Visualization
Generates Confusion Matrices, QWK Scores, and Uncertainty Plots for Journal Publication
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score

# ============================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================
# 🚨 UPDATE THIS PATH to your actual test dataset (CSV or Excel)
TEST_DATA_PATH = "/kaggle/input/dataset/Gardner_test_silver.csv"  
IMG_FOLDER = "/kaggle/input/dataset/Images/Images"

# This should point to the exact file saved in the previous step
MODEL_PATH = "/kaggle/working/saved_models/swin_focal_champion/swin_focal_seed42_best.pth" 
OUTPUT_DIR = "/kaggle/working/evaluation_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
MC_SAMPLES = 20  # Number of forward passes for uncertainty estimation

NUM_CLASSES_EXP = 5  
NUM_CLASSES_ICM = 4  
NUM_CLASSES_TE = 4   

# Clinical Mappings for the Confusion Matrices
EXP_MAP = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5'}
ICM_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'ND'}
TE_MAP = {0: 'A', 1: 'B', 2: 'C', 3: 'ND'}

# ============================================================
# DATASET & MODEL (Re-declared to load properly)
# ============================================================
class TestGardnerDataset(Dataset):
    def __init__(self, data_path, img_folder, transform=None):
        print(f"Loading Test Data: {data_path}...")
        # Automatically handle CSV or Excel
        if data_path.endswith('.csv'):
            self.df = pd.read_csv(data_path, sep=';')
        else:
            self.df = pd.read_excel(data_path)
            
        self.img_folder = img_folder
        self.transform = transform
        self.targets = ["EXP_silver", "ICM_silver", "TE_silver"]
        
        for col in self.targets:
            valid_mask = (self.df[col].notna()) & (self.df[col] != 'ND') & (self.df[col] != 'NA')
            self.df = self.df[valid_mask].copy()
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').astype(int)
            self.df = self.df[self.df[col].notna()].copy()
            
        print(f"Total test samples: {len(self.df)}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row['Image'])
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, row["EXP_silver"], row["ICM_silver"], row["TE_silver"]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class MultiTaskSwinWithUncertainty(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = swin_t(weights=None) # Don't need pretrained weights for evaluation
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        self.exp_head = self._make_head(in_features, NUM_CLASSES_EXP, dropout_rate)
        self.icm_head = self._make_head(in_features, NUM_CLASSES_ICM, dropout_rate)
        self.te_head = self._make_head(in_features, NUM_CLASSES_TE, dropout_rate)

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
# EVALUATION & UNCERTAINTY LOGIC
# ============================================================
def evaluate_mc_dropout(model, dataloader):
    model.eval()
    model.enable_dropout() # Turn dropout ON for inference
    
    results = {
        'exp': {'true': [], 'pred': [], 'entropy': []},
        'icm': {'true': [], 'pred': [], 'entropy': []},
        'te':  {'true': [], 'pred': [], 'entropy': []}
    }
    
    with torch.no_grad():
        for images, l_exp, l_icm, l_te in tqdm(dataloader, desc="Running MC Dropout"):
            images = images.to(DEVICE)
            
            # Store predictions for all MC samples
            batch_probs_exp = []
            batch_probs_icm = []
            batch_probs_te = []
            
            for _ in range(MC_SAMPLES):
                o_exp, o_icm, o_te = model(images)
                batch_probs_exp.append(torch.softmax(o_exp, dim=1))
                batch_probs_icm.append(torch.softmax(o_icm, dim=1))
                batch_probs_te.append(torch.softmax(o_te, dim=1))
                
            # Calculate Mean and Entropy for EXP
            mean_exp = torch.stack(batch_probs_exp).mean(dim=0)
            ent_exp = -torch.sum(mean_exp * torch.log(mean_exp + 1e-10), dim=1)
            pred_exp = torch.argmax(mean_exp, dim=1)
            
            # Calculate Mean and Entropy for ICM
            mean_icm = torch.stack(batch_probs_icm).mean(dim=0)
            ent_icm = -torch.sum(mean_icm * torch.log(mean_icm + 1e-10), dim=1)
            pred_icm = torch.argmax(mean_icm, dim=1)
            
            # Calculate Mean and Entropy for TE
            mean_te = torch.stack(batch_probs_te).mean(dim=0)
            ent_te = -torch.sum(mean_te * torch.log(mean_te + 1e-10), dim=1)
            pred_te = torch.argmax(mean_te, dim=1)
            
            # Save results
            results['exp']['true'].extend(l_exp.cpu().numpy())
            results['exp']['pred'].extend(pred_exp.cpu().numpy())
            results['exp']['entropy'].extend(ent_exp.cpu().numpy())
            
            results['icm']['true'].extend(l_icm.cpu().numpy())
            results['icm']['pred'].extend(pred_icm.cpu().numpy())
            results['icm']['entropy'].extend(ent_icm.cpu().numpy())
            
            results['te']['true'].extend(l_te.cpu().numpy())
            results['te']['pred'].extend(pred_te.cpu().numpy())
            results['te']['entropy'].extend(ent_te.cpu().numpy())
            
    return results

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================
def plot_clinical_confusion_matrix(y_true, y_pred, mapping, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    labels = [mapping[i] for i in range(len(mapping))]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Clinical Grade')
    plt.xlabel('AI Predicted Grade')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_uncertainty_histogram(y_true, y_pred, entropies, title, filename):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    entropies = np.array(entropies)
    
    correct_mask = (y_true == y_pred)
    incorrect_mask = (y_true != y_pred)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(entropies[correct_mask], color='green', alpha=0.5, label='Correct Predictions', stat='density', bins=20)
    sns.histplot(entropies[incorrect_mask], color='red', alpha=0.5, label='Incorrect Predictions', stat='density', bins=20)
    plt.title(f'MC Dropout Uncertainty (Entropy) Distribution: {title}')
    plt.xlabel('Predictive Entropy (Higher = More Uncertain)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# ============================================================
# MAIN EXECUTION
# ============================================================
def main():
    print("Loading Model...")
    model = MultiTaskSwinWithUncertainty().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    print("Loading Test Data...")
    test_ds = TestGardnerDataset(TEST_DATA_PATH, IMG_FOLDER, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Running MC Dropout Inference ({MC_SAMPLES} passes per image)...")
    res = evaluate_mc_dropout(model, test_loader)
    
    print("\n" + "="*50)
    print("FINAL CLINICAL METRICS (JOURNAL READY)")
    print("="*50)
    
    for task, mapping, name in [('exp', EXP_MAP, 'Expansion'), ('icm', ICM_MAP, 'ICM'), ('te', TE_MAP, 'TE')]:
        y_true = res[task]['true']
        y_pred = res[task]['pred']
        
        acc = accuracy_score(y_true, y_pred) * 100
        # Quadratic Weighted Kappa - The Gold Standard for Ordinal Medical Grading
        qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
        
        print(f"\n{name.upper()} RESULTS:")
        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Quadratic Weighted Kappa (QWK): {qwk:.4f}")
        
        # Generate and save plots
        plot_clinical_confusion_matrix(y_true, y_pred, mapping, name, f'cm_{name.lower()}.png')
        plot_uncertainty_histogram(y_true, y_pred, res[task]['entropy'], name, f'uncertainty_{name.lower()}.png')
        
    print(f"\n✅ All graphs saved successfully to: {OUTPUT_DIR}")
    print("Check the Kaggle file explorer on the right to download your .png files for your paper!")

if __name__ == "__main__":
    main()
