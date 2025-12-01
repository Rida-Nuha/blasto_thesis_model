import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# --- Configuration and Paths ---

SILVER_PATH = "data/Gardner_train_silver.csv"
GOLD_PATH   = "data/Gardner_test_gold.xlsx"
METRICS_FILE = "training_metrics.csv" # Matches the output file from main.py
IMG_FOLDER = "data/images" # Defined here for consistency

# Assuming dataset.py is in the same directory for imports
# (This ensures the robust parsing logic is available)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import parse_agreement 


# ----------------------------------------------------------------------------------
#  HELPER FUNCTION (Adapted from previous chat for robustness)
# ----------------------------------------------------------------------------------

def find_agreement_column(df, category="ICM"):
    """Finds the correct agreement column name based on a list of possibilities."""
    possible_cols = [
        f"{category}_Agreement", f"{category}_Agreement_", f"{category} agreement", 
        f"{category}_agreement", f"{category} Agreement", f"{category}_Agreement_des"
    ]
    
    # Check for specific column names used in the original inspector
    if category == "ICM" and "ICM agreement" in df.columns:
        return "ICM agreement"
    
    for col in possible_cols:
        if col in df.columns:
            return col
            
    return None 


# ----------------------------------------------------------------------------------
#  DATA LOADING AND PREPARATION
# ----------------------------------------------------------------------------------

def load_data():
    """Loads SILVER and GOLD data, merges and parses agreement scores."""
    print("Loading datasets...")
    silver = pd.read_csv(SILVER_PATH, sep=';')
    gold = pd.read_excel(GOLD_PATH)

    print(f"Silver samples: {len(silver)}")
    print(f"Gold samples:   {len(gold)}")

    # 1. FIND AND PARSE ICM AGREEMENT COLUMN from GOLD
    icm_col_name = find_agreement_column(gold, category="ICM")
    if icm_col_name is None:
        print("Warning: ICM agreement column not found in GOLD for agreement check.")
        gold['ICM_agreement_parsed'] = 1.0
    else:
        # Apply the robust parser function
        gold['ICM_agreement_parsed'] = gold[icm_col_name].apply(parse_agreement)


    # 2. MERGE AGREEMENT SCORES into SILVER
    merged = silver.merge(
        gold[["Image", 'ICM_agreement_parsed']], 
        on="Image",
        how="left"
    )

    # 3. FILL MISSING AGREEMENT (for silver samples without gold agreement score)
    merged['ICM_agreement_parsed'] = merged['ICM_agreement_parsed'].fillna(1.0)

    print("Merged dataset preview (SILVER + AGREEMENT):\n")
    print(merged.head())

    return merged


# ----------------------------------------------------------------------------------
#  VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------------------

def plot_icm_distribution(df):
    """Plot distribution of ICM labels (A/B/C or 0/1/2) from silver training."""
    plt.figure(figsize=(6,4))
    # Note: Assumes ICM_silver contains discrete grades (A/B/C or 0/1/2)
    sns.countplot(data=df, x="ICM_silver", order=df["ICM_silver"].value_counts().index.tolist())
    plt.title("ICM Grade Distribution (Silver Training Set)")
    plt.xlabel("ICM Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_agreement(df):
    """Plot agreement histogram after merging."""
    plt.figure(figsize=(6,4))
    # Use the newly created and cleaned column
    sns.histplot(df["ICM_agreement_parsed"], kde=True, bins=20, color='skyblue') 
    plt.title("Annotator Agreement Distribution for ICM")
    plt.xlabel("Agreement score (0–1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def plot_training_history():
    """Plots the loss and accuracy curves from the training_metrics.csv file."""
    if not os.path.exists(METRICS_FILE):
        print(f"\n❌ Training history file not found: {METRICS_FILE}. Run main.py first.")
        return

    print(f"\nLoading training history from {METRICS_FILE}...")
    metrics_df = pd.read_csv(METRICS_FILE)

    plt.figure(figsize=(12, 5))
    
    # --- Plot Loss ---
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss', marker='o')
    plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Soft Cross-Entropy)')
    plt.legend()
    plt.grid(True)
    
    # --- Plot Accuracy ---
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Validation Accuracy', marker='o', color='forestgreen')
    plt.title('Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    


# ----------------------------------------------------------------------------------
#  VERIFICATION FUNCTION
# ----------------------------------------------------------------------------------

def inspect_missing_images(df):
    """Check if images listed in CSV actually exist in the folder."""
    if not os.path.isdir(IMG_FOLDER):
        print(f"❌ Image folder not found at: {IMG_FOLDER}. Cannot check for missing images.")
        return

    missing = []
    unique_images = df["Image"].unique() 
    
    for filename in unique_images:
        if not os.path.exists(os.path.join(IMG_FOLDER, filename)):
            missing.append(filename)

    if len(missing) == 0:
        print("✅ All unique images exist. No missing files.")
    else:
        print(f"❌ Missing {len(missing)} images (out of {len(unique_images)} unique images):")
        for m in missing[:5]: 
            print(" -", m)
        if len(missing) > 5:
            print(" - ...and more.")


# ----------------------------------------------------------------------------------
#  MAIN EXECUTION
# ----------------------------------------------------------------------------------

def run_inspector():
    print("--- 📊 Data Sanity Check ---")
    try:
        df = load_data()

        print("\nChecking label balance...")
        plot_icm_distribution(df)

        print("\nChecking annotator agreement distribution...")
        plot_agreement(df)

        print("\nVerifying image files...")
        inspect_missing_images(df)
        
    except ValueError as e:
        print(f"An error occurred during data inspection: {e}")
    except FileNotFoundError as e:
        print(f"Error: One or more data files not found. Check your paths: {e}")

    print("\n--- 📈 Training History Check ---")
    plot_training_history()

    print("\nInspection complete! 🎉")


if __name__ == "__main__":
    run_inspector()