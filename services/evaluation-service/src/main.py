import hashlib
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
from PIL import Image, ImageOps
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
# DATASET PATHS
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"


# ============================================================
# MODEL / REPORT PATHS
# ============================================================

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

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

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

TRAINING_RUN_ID = os.getenv(
    "TRAINING_RUN_ID"
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

if not TRAINING_RUN_ID:
    raise RuntimeError(
        "TRAINING_RUN_ID is required."
    )


# MLflow authentication for DagsHub
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
# HASHING
# ============================================================

def calculate_image_hash(
    path: Path,
) -> str:
    """
    Calculate MD5 hash of the original image bytes.

    This is used to detect exact train/test duplicate
    images, matching the research evaluation protocol.
    """

    hash_md5 = hashlib.md5()

    with path.open("rb") as file:

        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


# ============================================================
# TRAIN IMAGE HASHES
# ============================================================

def create_train_hashes() -> set[str]:
    """
    Calculate hashes for all original training images.
    """

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory does not exist: "
            f"{TRAIN_DIR}"
        )

    train_hashes = set()

    total_train_images = 0

    for class_name in sorted(
        CLASS_TO_INDEX.keys()
    ):

        class_dir = (
            TRAIN_DIR / class_name
        )

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing training class directory: "
                f"{class_dir}"
            )

        for image_path in class_dir.iterdir():

            if not (
                image_path.is_file()
                and image_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                continue

            image_hash = calculate_image_hash(
                image_path
            )

            train_hashes.add(
                image_hash
            )

            total_train_images += 1

    logger.info(
        "Training images hashed: %d",
        total_train_images,
    )

    logger.info(
        "Unique training image hashes: %d",
        len(train_hashes),
    )

    return train_hashes


# ============================================================
# TEST DATAFRAME
# ============================================================

def create_test_dataframe() -> pd.DataFrame:
    """
    Build the clean test dataframe.

    Exact duplicates that are also present in the
    training set are excluded.

    Expected:
        Original test images: 700
        Duplicates removed:   2
        Clean test images:    698
    """

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory does not exist: "
            f"{TEST_DIR}"
        )

    train_hashes = create_train_hashes()

    records = []

    total_test_images = 0
    duplicate_count = 0

    for class_name in sorted(
        CLASS_TO_INDEX.keys()
    ):

        class_dir = (
            TEST_DIR / class_name
        )

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing test class directory: "
                f"{class_dir}"
            )

        for image_path in class_dir.iterdir():

            if not (
                image_path.is_file()
                and image_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                continue

            total_test_images += 1

            image_hash = calculate_image_hash(
                image_path
            )

            if image_hash in train_hashes:

                duplicate_count += 1

                logger.warning(
                    "Excluding train/test duplicate: %s",
                    image_path,
                )

                continue

            records.append(
                {
                    "path": str(image_path),
                    "class": class_name,
                    "image_hash": image_hash,
                }
            )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "No clean test images found."
        )

    logger.info(
        "Original test images: %d",
        total_test_images,
    )

    logger.info(
        "Exact train/test duplicates removed: %d",
        duplicate_count,
    )

    logger.info(
        "Clean test images: %d",
        len(dataframe),
    )

    logger.info(
        "Clean test distribution:\n%s",
        dataframe["class"]
        .value_counts()
        .sort_index()
        .to_string(),
    )

    return dataframe


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
# RESIZE WITH PADDING
# ============================================================

class ResizeWithPadding:
    """
    Resize while preserving the original aspect ratio,
    then pad to a square canvas.

    Final output:
        224 x 224
    """

    def __init__(
        self,
        size: int,
        fill=0,
    ):
        self.size = size
        self.fill = fill

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Image dimensions must be greater than zero."
            )

        scale = min(
            self.size / width,
            self.size / height,
        )

        new_width = int(
            round(width * scale)
        )

        new_height = int(
            round(height * scale)
        )

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        pad_left = (
            self.size - new_width
        ) // 2

        pad_top = (
            self.size - new_height
        ) // 2

        pad_right = (
            self.size
            - new_width
            - pad_left
        )

        pad_bottom = (
            self.size
            - new_height
            - pad_top
        )

        image = ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ),
            fill=self.fill,
        )

        return image


resize_with_padding = ResizeWithPadding(
    IMAGE_SIZE
)


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose(
    [
        resize_with_padding,

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
            f"Model artifact not found: "
            f"{MODEL_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint
    )

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
    """
    Run inference against the clean test dataset.
    """

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
                labels.cpu()
                .numpy()
                .tolist()
            )

            y_pred.extend(
                predictions.cpu()
                .numpy()
                .tolist()
            )

    return (
        np.array(y_true),
        np.array(y_pred),
    )


# ============================================================
# METRICS
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
    test_sample_count: int,
    duplicate_count: int,
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
        f"Test Samples: "
        f"{test_sample_count}\n"
        f"Exact Train/Test Duplicates Removed: "
        f"{duplicate_count}\n\n"
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
# MAIN
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

        logger.info(
            "Training run ID: %s",
            TRAINING_RUN_ID,
        )

        # ----------------------------------------------------
        # BUILD CLEAN TEST SET
        # ----------------------------------------------------

        dataframe = create_test_dataframe()

        duplicate_count = (
            700 - len(dataframe)
        )

        logger.info(
            "Final evaluation sample count: %d",
            len(dataframe),
        )

        # ----------------------------------------------------
        # LOAD MODEL
        # ----------------------------------------------------

        model = load_model()

        # ----------------------------------------------------
        # RUN EVALUATION
        # ----------------------------------------------------

        y_true, y_pred = evaluate(
            model,
            dataframe,
        )

        # ----------------------------------------------------
        # GENERATE METRICS
        # ----------------------------------------------------

        metrics = generate_report(
            y_true,
            y_pred,
        )

        # ----------------------------------------------------
        # LOG RESULTS
        # ----------------------------------------------------

        logger.info(
            "Test Accuracy: %.4f",
            metrics["accuracy"],
        )

        logger.info(
            "Macro Precision: %.4f",
            metrics["macro_precision"],
        )

        logger.info(
            "Macro Recall: %.4f",
            metrics["macro_recall"],
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

        # ----------------------------------------------------
        # SAVE REPORTS
        # ----------------------------------------------------

        report_txt, confusion_csv = save_report(
            metrics=metrics,
            test_sample_count=len(dataframe),
            duplicate_count=duplicate_count,
        )

        logger.info(
            "Saved classification report: %s",
            report_txt,
        )

        logger.info(
            "Saved confusion matrix: %s",
            confusion_csv,
        )

        # ====================================================
        # MLFLOW EVALUATION RUN
        # ====================================================

        with mlflow.start_run(
            run_name="efficientnet-b0-evaluation"
        ):

            # ------------------------------------------------
            # LINEAGE
            # ------------------------------------------------

            mlflow.set_tag(
                "training_run_id",
                TRAINING_RUN_ID,
            )

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

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

            # ------------------------------------------------
            # PARAMETERS
            # ------------------------------------------------

            mlflow.log_params(
                {
                    "model":
                        "efficientnet_b0",
                    "original_test_samples":
                        len(dataframe)
                        + duplicate_count,
                    "clean_test_samples":
                        len(dataframe),
                    "duplicate_test_samples_removed":
                        duplicate_count,
                    "device":
                        str(device),
                    "training_run_id":
                        TRAINING_RUN_ID,
                    "resize_strategy":
                        "resize_with_padding",
                    "image_size":
                        IMAGE_SIZE,
                }
            )

            # ------------------------------------------------
            # ARTIFACTS
            # ------------------------------------------------

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