"""
Multi-Task Swin Transformer Grad-CAM Visualization
Generates Explainable AI (XAI) Attention Heatmaps for the Thesis
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Import the grad-cam library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ============================================================
# 1. CONFIGURATION
# ============================================================
TEST_DATA_PATH = "/kaggle/input/datasets/ridakhan09/dataset/Gardner_test_gold.xlsx"
IMG_FOLDER = "/kaggle/input/datasets/ridakhan09/dataset/Images/Images"
MODEL_PATH = "/kaggle/working/saved_models/swin_champion_baseline/multitask_seed42_best.pth"
SAVE_DIR = "/kaggle/working/gradcam_visualizations"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# 2. MODEL ARCHITECTURE & WRAPPER
# ============================================================
class MultiTaskSwinWithUncertainty(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.backbone = swin_t(weights=None)
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        
        self.exp_head = self._make_head(in_features, 5, dropout_rate)
        self.icm_head = self._make_head(in_features, 3, dropout_rate)
        self.te_head = self._make_head(in_features, 3, dropout_rate)

    def _make_head(self, in_features, out_features, dropout_rate):
        return nn.Sequential(
            nn.LayerNorm(in_features), nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 256), nn.GELU(),
            nn.LayerNorm(256), nn.Dropout(p=dropout_rate), nn.Linear(256, out_features)
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.exp_head(features), self.icm_head(features), self.te_head(features)

# 🚨 WRAPPER: Isolates one task at a time for Grad-CAM to compute gradients
class TaskWrapper(nn.Module):
    def __init__(self, model, task="EXP"):
        super().__init__()
        self.model = model
        self.task = task
        
    def forward(self, x):
        out_exp, out_icm, out_te = self.model(x)
        if self.task == "EXP": return out_exp
        elif self.task == "ICM": return out_icm
        elif self.task == "TE": return out_te

# ============================================================
# 3. DATA & TRANSFORMS
# ============================================================
# We need two versions of the image: one normalized for the model, one raw for plotting
model_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

plot_transform = transforms.Compose([
    transforms.Resize((224, 224))
])

# Load just the first few images from the test set for visualization
df = pd.read_excel(TEST_DATA_PATH)
df = df.dropna(subset=["EXP_gold", "ICM_gold", "TE_gold"])
sample_images = df['Image'].head(5).tolist()

# ============================================================
# 4. GRAD-CAM GENERATION
# ============================================================

# 🚨 THE FIX: Translate the Swin-T matrix shape for standard Grad-CAM
def reshape_transform(tensor):
    # Swin outputs [Batch, Height, Width, Channels]
    # Grad-CAM expects [Batch, Channels, Height, Width]
    result = tensor.permute(0, 3, 1, 2)
    return result

def generate_heatmaps():
    print("Loading Champion Model...")
    base_model = MultiTaskSwinWithUncertainty().to(DEVICE)
    base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    base_model.eval()

    # Target the final feature block of the Swin Transformer
    target_layers = [base_model.backbone.features[-1]]

    tasks = ["EXP", "ICM", "TE"]
    
    for img_name in sample_images:
        img_path = os.path.join(IMG_FOLDER, str(img_name))
        if not os.path.exists(img_path): continue
            
        raw_pil = Image.open(img_path).convert('RGB')
        img_plot = np.float32(plot_transform(raw_pil)) / 255.0
        input_tensor = model_transform(raw_pil).unsqueeze(0).to(DEVICE)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_plot)
        axes[0].set_title(f"Original: {img_name}")
        axes[0].axis('off')

        for idx, task in enumerate(tasks):
            wrapped_model = TaskWrapper(base_model, task=task).to(DEVICE)
            wrapped_model.eval()
            
            # 🚨 Apply the reshape_transform here!
            cam = GradCAM(model=wrapped_model, 
                          target_layers=target_layers, 
                          reshape_transform=reshape_transform)
            
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
            visualization = show_cam_on_image(img_plot, grayscale_cam, use_rgb=True)
            
            axes[idx+1].imshow(visualization)
            axes[idx+1].set_title(f"{task} Attention Map")
            axes[idx+1].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(SAVE_DIR, f"gradcam_{img_name.split('.')[0]}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved visualization for {img_name}")

if __name__ == "__main__":
    generate_heatmaps()
