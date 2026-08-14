import logging
import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")

if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(PROJECT_ROOT_ENV)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset"
)

TEST_DIR = DATASET_DIR / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "best_efficientnet_b0.pth"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "evaluation"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3

CLASS_TO_INDEX = {
    "glioma": 0,
    "meningioma": 1,
    "no_tumor": 2,
}

INDEX_TO_CLASS = {
    0: "glioma",
    1: "meningioma",
    2: "no_tumor",
}


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "brainlens-training",
)

DAGSHUB_USERNAME = os.getenv(
    "DAGSHUB_USERNAME"
)

DAGSHUB_TOKEN = os.getenv(
    "DAGSHUB_TOKEN"
)

if not MLFLOW_TRACKING_URI:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI is required."
    )

if not DAGSHUB_USERNAME:
    raise RuntimeError(
        "DAGSHUB_USERNAME is required."
    )

if not DAGSHUB_TOKEN:
    raise RuntimeError(
        "DAGSHUB_TOKEN is required."
    )

os.environ["MLFLOW_TRACKING_USERNAME"] = (
    DAGSHUB_USERNAME
)

os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    DAGSHUB_TOKEN
)

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

logger.info(
    "Using device: %s",
    device,
)


# ============================================================
# DATASET
# ============================================================

class BrainTumorTestDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(
            drop=True
        )
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = row["path"]
        class_name = row["class"]

        image = Image.open(
            image_path
        ).convert("RGB")

        label = torch.tensor(
            CLASS_TO_INDEX[class_name],
            dtype=torch.long,
        )

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# ============================================================
# BUILD TEST DATAFRAME
# ============================================================

def create_test_dataframe() -> pd.DataFrame:

    records = []

    for class_name in sorted(
        CLASS_TO_INDEX.keys()
    ):

        class_dir = TEST_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }
            ):
                records.append(
                    {
                        "path": str(image_path),
                        "class": class_name,
                    }
                )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "No test images found."
        )

    return dataframe


# ============================================================
# TRANSFORMS
# ============================================================

test_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MODEL
# ============================================================

def create_model() -> nn.Module:

    model = models.efficientnet_b0(
        weights=None
    )

    num_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(
            p=0.30
        ),
        nn.Linear(
            num_features,
            NUM_CLASSES,
        ),
    )

    return model


def load_model() -> nn.Module:

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    logger.info(
        "Loaded model: %s",
        MODEL_PATH,
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model: nn.Module,
    dataframe: pd.DataFrame,
):
    dataset = BrainTumorTestDataset(
        dataframe,
        transform=test_transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy().tolist()
            )

            y_pred.extend(
                predictions.cpu().numpy().tolist()
            )

    return (
        np.array(y_true),
        np.array(y_pred),
    )


# ============================================================
# REPORTING
# ============================================================

def generate_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
):
    labels = [
        CLASS_TO_INDEX["glioma"],
        CLASS_TO_INDEX["meningioma"],
        CLASS_TO_INDEX["no_tumor"],
    ]

    target_names = [
        "glioma",
        "meningioma",
        "no_tumor",
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    metrics: dict,
) -> tuple[Path, Path]:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_txt = (
        REPORT_DIR
        / "classification_report.txt"
    )

    confusion_csv = (
        REPORT_DIR
        / "confusion_matrix.csv"
    )

    report_text = (
        f"Accuracy: "
        f"{metrics['accuracy']:.4f}\n"
        f"Macro Precision: "
        f"{metrics['macro_precision']:.4f}\n"
        f"Macro Recall: "
        f"{metrics['macro_recall']:.4f}\n"
        f"Macro F1: "
        f"{metrics['macro_f1']:.4f}\n"
        f"Weighted F1: "
        f"{metrics['weighted_f1']:.4f}\n\n"
        f"Classification Report:\n\n"
        f"{metrics['classification_report']}\n"
    )

    report_txt.write_text(
        report_text,
        encoding="utf-8",
    )

    confusion_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[
            "actual_glioma",
            "actual_meningioma",
            "actual_no_tumor",
        ],
        columns=[
            "pred_glioma",
            "pred_meningioma",
            "pred_no_tumor",
        ],
    )

    confusion_df.to_csv(
        confusion_csv
    )

    return (
        report_txt,
        confusion_csv,
    )


# ============================================================
# MAIN EVALUATION FLOW
# ============================================================

def main() -> None:

    try:

        logger.info(
            "Starting evaluation service..."
        )

        logger.info(
            "Test dataset: %s",
            TEST_DIR,
        )

        dataframe = create_test_dataframe()

        logger.info(
            "Total test images: %d",
            len(dataframe),
        )

        model = load_model()

        y_true, y_pred = evaluate(
            model,
            dataframe,
        )

        metrics = generate_report(
            y_true,
            y_pred,
        )

        logger.info(
            "Test Accuracy: %.4f",
            metrics["accuracy"],
        )

        logger.info(
            "Macro F1: %.4f",
            metrics["macro_f1"],
        )

        logger.info(
            "Weighted F1: %.4f",
            metrics["weighted_f1"],
        )

        logger.info(
            "\nClassification Report:\n%s",
            metrics["classification_report"],
        )

        logger.info(
            "\nConfusion Matrix:\n%s",
            metrics["confusion_matrix"],
        )

        report_txt, confusion_csv = save_report(
            metrics
        )

        logger.info(
            "Saved classification report: %s",
            report_txt,
        )

        logger.info(
            "Saved confusion matrix: %s",
            confusion_csv,
        )

        # ----------------------------------------------------
        # MLFLOW EVALUATION RUN
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name="efficientnet-b0-evaluation"
        ):

            mlflow.log_metrics(
                {
                    "test_accuracy":
                        metrics["accuracy"],
                    "test_macro_precision":
                        metrics["macro_precision"],
                    "test_macro_recall":
                        metrics["macro_recall"],
                    "test_macro_f1":
                        metrics["macro_f1"],
                    "test_weighted_f1":
                        metrics["weighted_f1"],
                }
            )

            mlflow.log_params(
                {
                    "model":
                        "efficientnet_b0",
                    "test_samples":
                        len(dataframe),
                    "device":
                        str(device),
                }
            )

            mlflow.log_artifact(
                str(report_txt),
                artifact_path="evaluation",
            )

            mlflow.log_artifact(
                str(confusion_csv),
                artifact_path="evaluation",
            )

        logger.info(
            "Evaluation completed successfully."
        )

        sys.exit(0)

    except Exception:

        logger.exception(
            "Evaluation service failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()