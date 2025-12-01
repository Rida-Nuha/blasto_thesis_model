import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from dataset import BlastocystDataset, val_transform
from torchvision.models import swin_t, Swin_T_Weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def confusion_matrix_eval():
    SILVER_PATH = "data/Gardner_train_silver.csv"
    GOLD_PATH = "data/Gardner_test_gold.xlsx"
    IMG_FOLDER = "data/images"

    dataset = BlastocystDataset(
        silver_csv=SILVER_PATH,
        gold_xlsx=GOLD_PATH,
        root_dir=IMG_FOLDER,
        transform=val_transform
    )

    _, val_ds = random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    model.head = nn.Linear(model.head.in_features, 3)
    model.load_state_dict(torch.load("saved_models/swin_transformer_icm.pth"))
    model.to(DEVICE)
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for imgs, soft_labels, _ in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            predictions = torch.argmax(outputs, 1).cpu().numpy()
            labels = torch.argmax(soft_labels, 1).cpu().numpy()

            preds.extend(predictions)
            gts.extend(labels)

    cm = confusion_matrix(gts, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["A", "B", "C"])
    disp.plot(cmap="Blues")
    plt.title("ICM Confusion Matrix (Validation Set)")
    plt.savefig("confusion_matrix_icm.png")
    plt.show()

if __name__ == "__main__":
    confusion_matrix_eval()
