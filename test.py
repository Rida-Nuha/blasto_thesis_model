import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm # For progress bar

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 3

# --- Configuration and Paths ---
OUTPUT_FILENAME = "swin_icm_predictions_with_probs.csv"
MODEL_PATH = "saved_models/swin_transformer_icm.pth" # Standard path from main.py
BEST_CHECKPOINT_PATH = "saved_models/best_swin_icm.pth" # Best checkpoint from EarlyStopping

# Same normalization and resize as validation transform (val_transform in dataset.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Label decoding (Class indices 0, 1, 2 map to A, B, C)
decode = {0: "A", 1: "B", 2: "C"}


def load_model():
    """Initializes the Swin-T model and loads the best saved weights."""
    print("Loading model and weights...")
    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    
    # Check for the best checkpoint (created by EarlyStopping) first
    weights_to_load = BEST_CHECKPOINT_PATH if os.path.exists(BEST_CHECKPOINT_PATH) else MODEL_PATH
    
    # To correctly load the checkpoint, we must reconstruct the model architecture exactly, 
    # including the Dropout layer we added in main.py.
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=0.5), # Ensure the Dropout rate matches main.py (p=0.5)
        nn.Linear(in_features, NUM_CLASSES)
    )
    
    if not os.path.exists(weights_to_load):
        raise FileNotFoundError(f"Model weights not found at: {weights_to_load}. Run main.py first!")
        
    model.load_state_dict(torch.load(weights_to_load, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print(f"Loaded weights from: {weights_to_load}")
    return model


def predict_folder(image_folder):
    """
    Performs batch prediction on all images in a folder and saves results 
    including class probabilities.
    """
    if not os.path.isdir(image_folder):
        print(f"Error: Image folder not found at '{image_folder}'")
        return

    try:
        model = load_model()
    except FileNotFoundError as e:
        print(e)
        return

    results = []
    
    # 1. Get all image file names
    all_files = [f for f in os.listdir(image_folder) 
                 if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    print(f"\nStarting prediction for {len(all_files)} images...")

    # 2. Iterate with tqdm for a progress bar
    for fname in tqdm(all_files, desc="Predicting Blastocysts"):
        path = os.path.join(image_folder, fname)

        try:
            img = Image.open(path).convert("RGB")
            # Preprocess: transform, add batch dimension (unsqueeze), and move to device
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        except Exception as e:
            print(f"Error processing image {fname}: {e}. Skipping.")
            continue

        with torch.no_grad():
            out = model(img_tensor)
            
            # Apply Softmax to get probabilities (confidence scores)
            probabilities = torch.softmax(out, dim=1).cpu().squeeze()
            
            # Get the hard class prediction index
            pred_idx = torch.argmax(probabilities).item()

        # Format probabilities for the results list
        probs_list = probabilities.tolist()
        
        results.append([
            fname, 
            decode[pred_idx], 
            probs_list[0], # Prob_A (Class 0)
            probs_list[1], # Prob_B (Class 1)
            probs_list[2]  # Prob_C (Class 2)
        ])

    # 3. Create DataFrame and save
    df = pd.DataFrame(
        results, 
        columns=["Image", "Predicted_ICM", "Prob_A", "Prob_B", "Prob_C"]
    )
    
    df.to_csv(OUTPUT_FILENAME, index=False)
    print(f"\n🎉 Predictions complete and results saved to → {OUTPUT_FILENAME}")


if __name__ == "__main__":
    predict_folder("data/images")