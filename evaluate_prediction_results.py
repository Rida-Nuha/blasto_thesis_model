import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, average_precision_score

# --- Configuration and Paths ---
GOLD_PATH = "data/Gardner_test_gold.xlsx"
PREDICTIONS_FILE = "swin_icm_predictions_with_probs.csv"

# Label decoding: REVERSED MAPPING (Hypothesis: Model output 0 = Worst ICM (C), 2 = Best ICM (A))
LABEL_MAP = {
    0: "C", 
    1: "B", 
    2: "A"
}
# TARGET_NAMES must be listed in order of their index 0, 1, 2 for the Confusion Matrix
TARGET_NAMES = ["C", "B", "A"] 

# --- Helper Functions ---

def parse_gold_labels(gold_df):
    """
    Extracts and maps the 'ICM_gold' numeric labels (0, 1, 2) 
    from the gold file using the revised mapping (0=C, 2=A).
    """
    # Filter out samples where ICM_gold is not one of the target indices (0, 1, or 2)
    valid_labels = gold_df['ICM_gold'].isin(LABEL_MAP.keys())
    
    # Map the numeric indices to string labels ("C", "B", "A")
    gold_df['True_ICM'] = gold_df['ICM_gold'].apply(lambda x: LABEL_MAP.get(x, 'Invalid'))
    
    return gold_df[valid_labels].reset_index(drop=True)

def calculate_binary_metrics(y_true, y_score):
    """Calculates AUROC and AUPRC for a single binary task."""
    # Checks if both classes (0 and 1) are present; required for AUC calculation
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
        
    try:
        auroc = roc_auc_score(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        return auroc, auprc
    except ValueError:
        return np.nan, np.nan

def report_binary_metrics(df):
    """Reports AUROC and AUPRC for the One-vs-Rest tasks."""
    print("\n--- 📈 AUROC and AUPRC (One-vs-Rest) ---")
    
    # Probabilities must be read in the order they were predicted: Prob_C, Prob_B, Prob_A
    y_probs = df[['Prob_C', 'Prob_B', 'Prob_A']].values # Adjusted column order for OvR calculation
    results = {}
    
    # Task 1: C (Poor) vs. Rest (B/A) -> Uses index 0
    y_true_C = (df['True_ICM'] == 'C').astype(int)
    y_score_C = y_probs[:, 0] # Probability of C
    auroc_C, auprc_C = calculate_binary_metrics(y_true_C, y_score_C)
    results['C_vs_Rest'] = {'AUROC': auroc_C, 'AUPRC': auprc_C}

    # Task 2: B (Average) vs. Rest (C/A) -> Uses index 1
    y_true_B = (df['True_ICM'] == 'B').astype(int)
    y_score_B = y_probs[:, 1] # Probability of B
    auroc_B, auprc_B = calculate_binary_metrics(y_true_B, y_score_B)
    results['B_vs_Rest'] = {'AUROC': auroc_B, 'AUPRC': auprc_B}

    # Task 3: A (Good) vs. Rest (C/B) -> Uses index 2
    y_true_A = (df['True_ICM'] == 'A').astype(int)
    y_score_A = y_probs[:, 2] # Probability of A
    auroc_A, auprc_A = calculate_binary_metrics(y_true_A, y_score_A)
    results['A_vs_Rest'] = {'AUROC': auroc_A, 'AUPRC': auprc_A}

    # Print Results
    metrics_df = pd.DataFrame.from_dict(results, orient='index')
    print(metrics_df.round(4))


# ---------------------------------------------------------
#  MAIN EVALUATION FUNCTION
# ---------------------------------------------------------

def evaluate_model():
    print("--- 🔬 Model Evaluation Start ---")

    # 1. Load Data
    try:
        pred_df = pd.read_csv(PREDICTIONS_FILE)
        gold_df = pd.read_excel(GOLD_PATH)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Ensure {PREDICTIONS_FILE} and {GOLD_PATH} exist.")
        print(e)
        return

    # 2. Process Gold Labels and Merge
    true_df = parse_gold_labels(gold_df)
    evaluation_df = pd.merge(
        pred_df, 
        true_df[['Image', 'True_ICM']], 
        on='Image', 
        how='inner'
    )
    
    if evaluation_df.empty:
        print("Error: No overlapping images found between predictions and gold standard.")
        return

    print(f"Evaluated {len(evaluation_df)} matched samples.")

    # 3. Hard-Label Metrics (Confusion Matrix, F1, Precision, Recall)
    y_true_hard = evaluation_df['True_ICM']
    y_pred_hard = evaluation_df['Predicted_ICM']
    
    # Calculate and print Confusion Matrix 
    cm = confusion_matrix(y_true_hard, y_pred_hard, labels=TARGET_NAMES)
    print("\n--- 📊 Confusion Matrix ---")
    # Note: Rows/Cols are now C, B, A based on TARGET_NAMES
    print("Rows: True Label (C, B, A) | Columns: Predicted Label (C, B, A)") 
    print(cm)
    
    # Calculate and print Classification Report
    report = classification_report(y_true_hard, y_pred_hard, target_names=TARGET_NAMES, zero_division=0)
    print("\n--- 📈 Classification Report (F1, Precision, Recall) ---")
    print(report)
    
    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
    plt.title('Confusion Matrix: True vs. Predicted ICM Grade')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 4. Probability-Based Metrics (AUROC / AUPRC)
    report_binary_metrics(evaluation_df)

    print("\nEvaluation complete. 🎉")

if __name__ == "__main__":
    evaluate_model()