"""
Test Dataset Loader for Gardner Gold Standard
Loads test images with ground truth labels from Gardner_test_gold.xlsx
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


# ============================================================
# TRANSFORMS (Same as training)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# TEST DATASET
# ============================================================
class BlastocystTestDataset(Dataset):
    """
    Test dataset from Gardner gold standard.
    Loads ICM_gold as ground truth labels.
    """
    
    def __init__(self, gold_xlsx, root_dir, transform=None):
        """
        Args:
            gold_xlsx: Path to Gardner_test_gold.xlsx
            root_dir: Directory with images
            transform: Transform to apply
        """
        self.data_frame = pd.read_excel(gold_xlsx)
        self.root_dir = root_dir
        self.transform = transform
        
        # Clean data
        self._clean_data()
        self._validate_images()
        self._print_statistics()
    
    def _clean_data(self):
        """Clean and filter test data"""
        # Normalize image names
        self.data_frame["Image"] = self.data_frame["Image"].astype(str).str.strip()
        
        # Map ICM_gold to class indices
        def map_icm_gold(value):
            """Map ICM gold label to class index"""
            if pd.isna(value) or value == 'ND' or value == 'NA':
                return None
            
            try:
                val = int(value)
            except:
                return None
            
            # Map: 0 → 0, 1 → 1, 3 → 2
            if val == 0:
                return 0
            elif val == 1:
                return 1
            elif val == 3:
                return 2
            else:
                return None
        
        # Apply mapping
        self.data_frame['ICM_class'] = self.data_frame['ICM_gold'].apply(map_icm_gold)
        
        # Remove rows with invalid labels (ND, NA, etc.)
        before = len(self.data_frame)
        self.data_frame = self.data_frame[self.data_frame['ICM_class'].notna()].reset_index(drop=True)
        after = len(self.data_frame)
        
        if before > after:
            print(f"⚠️  Removed {before - after} samples with invalid/missing labels (ND/NA)")
        
        # Parse agreement scores (optional, for analysis)
        def parse_agreement(value):
            if pd.isna(value):
                return None
            value = str(value).strip()
            if value in ["ND", "NA", "revised_cons", ""]:
                return None
            if "/" in value:
                try:
                    num, den = value.split("/")
                    return float(num) / float(den)
                except:
                    return None
            try:
                return float(value.replace(",", "."))
            except:
                return None
        
        # Try to find ICM agreement column
        icm_agree_cols = ['ICM_Agreement', 'ICM_Agreement_des']
        for col in icm_agree_cols:
            if col in self.data_frame.columns:
                self.data_frame['agreement'] = self.data_frame[col].apply(parse_agreement)
                break
        
        if 'agreement' not in self.data_frame.columns:
            self.data_frame['agreement'] = 1.0  # Default
    
    def _validate_images(self):
        """Check which images exist"""
        valid_indices = []
        missing = []
        
        for idx, row in self.data_frame.iterrows():
            img_path = os.path.join(self.root_dir, row["Image"])
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing.append(row["Image"])
        
        if missing:
            print(f"⚠️  {len(missing)} test images not found - removing from dataset")
            if len(missing) <= 10:
                print(f"   Missing: {missing}")
        
        self.data_frame = self.data_frame.loc[valid_indices].reset_index(drop=True)
    
    def _print_statistics(self):
        """Print test dataset statistics"""
        print("\n" + "="*70)
        print("TEST DATASET STATISTICS (Gold Standard)")
        print("="*70)
        print(f"Total test samples: {len(self.data_frame)}")
        
        # Class distribution
        class_counts = self.data_frame['ICM_class'].value_counts().sort_index()
        print("\nClass distribution:")
        class_names = {0: "Poor (0)", 1: "Medium (1)", 2: "Good (3)"}
        
        for cls in sorted(class_counts.index):
            count = class_counts[cls]
            pct = 100 * count / len(self.data_frame)
            print(f"  Class {int(cls)} [{class_names.get(cls, 'Unknown')}]: "
                  f"{count} samples ({pct:.1f}%)")
        
        # Agreement statistics (if available)
        if 'agreement' in self.data_frame.columns:
            agreements = self.data_frame['agreement'].dropna()
            if len(agreements) > 0:
                print(f"\nAgreement scores (n={len(agreements)}):")
                print(f"  Mean: {agreements.mean():.3f}")
                print(f"  Std:  {agreements.std():.3f}")
                print(f"  Min:  {agreements.min():.3f}")
                print(f"  Max:  {agreements.max():.3f}")
        
        print("="*70 + "\n")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: Ground truth class label (0, 1, or 2)
            info: Dict with additional info (image name, agreement, etc.)
        """
        row = self.data_frame.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Image"])
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Get label
        label = int(row['ICM_class'])
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Additional info
        info = {
            'image_name': row['Image'],
            'agreement': row.get('agreement', None)
        }
        
        return image, label, info


# ============================================================
# TESTING
# ============================================================
if __name__ == "__main__":
    """Test the dataset loader"""
    
    test_dataset = BlastocystTestDataset(
        gold_xlsx="data/Gardner_test_gold.xlsx",
        root_dir="data/images",
        transform=test_transform
    )
    
    print(f"\n✅ Test dataset loaded successfully!")
    print(f"   Total samples: {len(test_dataset)}")
    
    # Test loading a sample
    if len(test_dataset) > 0:
        img, label, info = test_dataset[0]
        print(f"\n✅ Sample loaded:")
        print(f"   Image shape: {img.shape}")
        print(f"   Label: {label}")
        print(f"   Image name: {info['image_name']}")
        print(f"   Agreement: {info['agreement']}")
