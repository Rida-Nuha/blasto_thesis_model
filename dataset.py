"""
Blastocyst Dataset - Fixed Version
Handles ICM grading with proper label mapping and class balancing
"""

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np


# ============================================================
# TRANSFORMS
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# DATASET
# ============================================================
class BlastocystDataset(Dataset):
    """
    Blastocyst ICM grading dataset.
    
    Handles label mapping from Gardner grading (0, 1, 3) to class indices (0, 1, 2).
    Note: Original data has grades 0, 1, 3 (no grade 2).
    """
    
    def __init__(self, csv_path, root_dir, transform=None):
        """
        Args:
            csv_path: Path to CSV with image names and labels
            root_dir: Directory with images
            transform: Optional transform to apply
        """
        self.data_frame = pd.read_csv(csv_path, sep=";")
        self.root_dir = root_dir
        self.transform = transform
        
        # Clean and validate data
        self._clean_data()
        self._validate_images()
        self._print_statistics()
        
    def _clean_data(self):
        """Clean and map labels correctly"""
        # Normalize image names
        self.data_frame["Image"] = self.data_frame["Image"].astype(str).str.strip()
        
        # Map ICM grades to class indices
        # Original grades: 0, 1, 3 → Classes: 0, 1, 2
        def map_icm_label(value):
            """Map ICM grade to class index"""
            # Handle different formats
            if pd.isna(value):
                return None
            
            # Convert to int if possible
            try:
                val = int(value)
            except:
                # Handle string labels like 'A', 'B', 'C'
                if str(value).upper() in ['A', '3']:
                    return 2  # Best quality → Class 2
                elif str(value).upper() in ['B', '1']:
                    return 1  # Medium quality → Class 1
                elif str(value).upper() in ['C', '0']:
                    return 0  # Poor quality → Class 0
                else:
                    return None
            
            # Map numeric grades
            if val == 3:
                return 2  # Grade 3 (best) → Class 2
            elif val == 1:
                return 1  # Grade 1 (medium) → Class 1
            elif val == 0:
                return 0  # Grade 0 (poor) → Class 0
            else:
                return None
        
        # Apply mapping
        self.data_frame['ICM_class'] = self.data_frame['ICM_silver'].apply(map_icm_label)
        
        # Remove rows with invalid labels
        before = len(self.data_frame)
        self.data_frame = self.data_frame[self.data_frame['ICM_class'].notna()].reset_index(drop=True)
        after = len(self.data_frame)
        
        if before > after:
            print(f"⚠️  Removed {before - after} samples with invalid labels")
    
    def _validate_images(self):
        """Check which images exist and filter dataset"""
        valid_indices = []
        missing = []
        
        for idx, row in self.data_frame.iterrows():
            img_path = os.path.join(self.root_dir, row["Image"])
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                missing.append(row["Image"])
        
        if missing:
            print(f"⚠️  {len(missing)} images not found - removing from dataset")
            if len(missing) <= 5:
                print(f"   Missing: {missing}")
            else:
                print(f"   Example missing: {missing[:5]}")
        
        self.data_frame = self.data_frame.loc[valid_indices].reset_index(drop=True)
    
    def _print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*70)
        print("BLASTOCYST DATASET STATISTICS")
        print("="*70)
        print(f"Total samples: {len(self.data_frame)}")
        
        # Class distribution
        class_counts = self.data_frame['ICM_class'].value_counts().sort_index()
        print("\nClass distribution:")
        class_names = {0: "Poor (0)", 1: "Medium (1)", 2: "Good (3)"}
        
        for cls in sorted(class_counts.index):
            count = class_counts[cls]
            pct = 100 * count / len(self.data_frame)
            print(f"  Class {int(cls)} [{class_names.get(cls, 'Unknown')}]: "
                  f"{count} samples ({pct:.1f}%)")
        
        # Class imbalance ratio
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"\n⚠️  Class imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 5:
            print("   → Severe imbalance! Using weighted sampling recommended.")
        
        print("="*70 + "\n")
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: Class label (0, 1, or 2)
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
        
        return image, label
    
    def get_class_weights(self):
        """
        Calculate class weights for handling imbalance.
        
        Returns:
            torch.Tensor: Weights for each class
        """
        class_counts = self.data_frame['ICM_class'].value_counts().sort_index()
        total = len(self.data_frame)
        
        # Inverse frequency weighting
        weights = torch.zeros(3)
        for cls in range(3):
            count = class_counts.get(cls, 1)  # Avoid division by zero
            weights[cls] = total / (3 * count)
        
        return weights
    
    def get_sample_weights(self):
        """
        Get per-sample weights for WeightedRandomSampler.
        
        Returns:
            list: Weight for each sample
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[int(label)] for label in self.data_frame['ICM_class']]
        return sample_weights


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_dataloaders(train_dataset, val_dataset, batch_size=16, num_workers=4):
    """
    Create train and validation dataloaders with proper sampling.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader, val_loader
    """
    from torch.utils.data import DataLoader, WeightedRandomSampler
    
    # Weighted sampler for training to handle class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    return train_loader, val_loader


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    """Test dataset loading"""
    
    # Load dataset
    dataset = BlastocystDataset(
        csv_path="data/Gardner_train_silver.csv",
        root_dir="data/images",
        transform=train_transform
    )
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"   Total samples: {len(dataset)}")
    
    # Test loading a sample
    img, label = dataset[0]
    print(f"\n✅ Sample loaded:")
    print(f"   Image shape: {img.shape}")
    print(f"   Label: {label}")
    print(f"   Label type: {type(label)}")
    
    # Show class weights
    weights = dataset.get_class_weights()
    print(f"\n📊 Class weights (for loss function):")
    print(f"   {weights}")