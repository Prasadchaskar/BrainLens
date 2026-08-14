import logging
import os
import random
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
from sklearn.model_selection import train_test_split
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
    / "processed"
    / "dataset"
)

TRAIN_DIR = DATASET_DIR / "train"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

NUM_CLASSES = 3

# Keep this at 2 for the MLflow smoke test.
# Change to 8 after MLflow is verified.
NUM_EPOCHS = 8

PATIENCE = 5

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4

BEST_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "best_efficientnet_b0.pth"
)


# ============================================================
# CLASS MAPPING
# ============================================================

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


# MLflow expects these names for HTTP authentication.
os.environ["MLFLOW_TRACKING_USERNAME"] = (
    DAGSHUB_USERNAME
)

os.environ["MLFLOW_TRACKING_PASSWORD"] = (
    DAGSHUB_TOKEN
)


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

def get_or_create_experiment(
    experiment_name: str
) -> str:
    """Return an active MLflow experiment ID."""

    logger.info(
        "Checking MLflow experiment: %s",
        experiment_name,
    )

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        logger.info(
            "Experiment does not exist. Creating: %s",
            experiment_name,
        )

        experiment_id = mlflow.create_experiment(
            experiment_name
        )

        logger.info(
            "Created MLflow experiment: %s",
            experiment_id,
        )

        return experiment_id

    if experiment.lifecycle_stage == "deleted":
        logger.info(
            "Experiment exists but is deleted. "
            "Restoring experiment: %s",
            experiment.experiment_id,
        )

        client = mlflow.tracking.MlflowClient()

        client.restore_experiment(
            experiment.experiment_id
        )

        logger.info(
            "Experiment restored: %s",
            experiment.experiment_id,
        )

        return experiment.experiment_id

    logger.info(
        "Using existing active MLflow experiment: %s",
        experiment.experiment_id,
    )

    return experiment.experiment_id

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

class BrainTumorDataset(Dataset):

    def __init__(
        self,
        dataframe,
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
# BUILD DATAFRAME
# ============================================================

def create_dataframe():

    records = []

    for class_name in sorted(
        CLASS_TO_INDEX.keys()
    ):

        class_dir = TRAIN_DIR / class_name

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
            "No training images found."
        )

    return dataframe


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        ),

        transforms.RandomAffine(
            degrees=5,
            translate=(0.02, 0.02),
            scale=(0.98, 1.02),
            fill=0,
        ),

        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
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


val_transform = transforms.Compose(
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

def create_model():

    model = models.efficientnet_b0(
        weights=(
            models.EfficientNet_B0_Weights.DEFAULT
        )
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

    model = model.to(device)

    for parameter in model.parameters():
        parameter.requires_grad = True

    return model


# ============================================================
# TRAINING
# ============================================================

def train():

    logger.info(
        "Starting training service..."
    )

    experiment_id = get_or_create_experiment(
        MLFLOW_EXPERIMENT_NAME
    )

    dataframe = create_dataframe()

    logger.info(
        "Total training images: %d",
        len(dataframe),
    )

    train_df, val_df = train_test_split(
        dataframe,
        test_size=0.20,
        random_state=SEED,
        stratify=dataframe["class"],
    )

    train_dataset = BrainTumorDataset(
        train_df,
        transform=train_transform,
    )

    val_dataset = BrainTumorDataset(
        val_df,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        "Training images: %d",
        len(train_dataset),
    )

    logger.info(
        "Validation images: %d",
        len(val_dataset),
    )

    model = create_model()

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.classifier.parameters(),
                "lr": CLASSIFIER_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    BEST_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    # ========================================================
    # MLFLOW RUN
    # ========================================================

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="efficientnet-b0-baseline"
    ):

        # ----------------------------------------------------
        # LOG PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model": "efficientnet_b0",
                "pretrained": True,
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "num_classes": NUM_CLASSES,
                "num_epochs": NUM_EPOCHS,
                "patience": PATIENCE,
                "backbone_lr": BACKBONE_LR,
                "classifier_lr": CLASSIFIER_LR,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": "AdamW",
                "loss": "CrossEntropyLoss",
                "seed": SEED,
                "train_samples": len(train_dataset),
                "validation_samples": len(val_dataset),
                "device": str(device),
            }
        )

        # ----------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------

        for epoch in range(NUM_EPOCHS):

            # =================================================
            # TRAIN
            # =================================================

            model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for images, labels in train_loader:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad()

                outputs = model(images)

                loss = loss_function(
                    outputs,
                    labels,
                )

                loss.backward()

                optimizer.step()

                train_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                train_correct += (
                    predictions == labels
                ).sum().item()

                train_total += labels.size(0)

            train_loss /= len(train_loader)

            train_accuracy = (
                train_correct
                / train_total
            )

            # =================================================
            # VALIDATION
            # =================================================

            model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for images, labels in val_loader:

                    images = images.to(
                        device,
                        non_blocking=True,
                    )

                    labels = labels.to(
                        device,
                        non_blocking=True,
                    )

                    outputs = model(images)

                    loss = loss_function(
                        outputs,
                        labels,
                    )

                    val_loss += loss.item()

                    predictions = torch.argmax(
                        outputs,
                        dim=1,
                    )

                    val_correct += (
                        predictions == labels
                    ).sum().item()

                    val_total += labels.size(0)

            val_loss /= len(val_loader)

            val_accuracy = (
                val_correct
                / val_total
            )

            # =================================================
            # LOG CONSOLE
            # =================================================

            logger.info(
                "Epoch [%d/%d] "
                "Train Loss: %.4f "
                "Train Acc: %.4f "
                "Val Loss: %.4f "
                "Val Acc: %.4f",
                epoch + 1,
                NUM_EPOCHS,
                train_loss,
                train_accuracy,
                val_loss,
                val_accuracy,
            )

            # =================================================
            # LOG METRICS TO MLFLOW
            # =================================================

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                },
                step=epoch + 1,
            )

            # =================================================
            # BEST MODEL
            # =================================================

            if val_accuracy > best_val_accuracy:

                best_val_accuracy = (
                    val_accuracy
                )

                best_epoch = epoch + 1

                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    BEST_MODEL_PATH,
                )

                mlflow.log_metric(
                    "best_val_accuracy",
                    best_val_accuracy,
                    step=best_epoch,
                )

                logger.info(
                    "New best model saved. "
                    "Validation accuracy: %.4f",
                    val_accuracy,
                )

            else:

                epochs_without_improvement += 1

                logger.info(
                    "No improvement for %d epoch(s). "
                    "Best validation accuracy: %.4f",
                    epochs_without_improvement,
                    best_val_accuracy,
                )

                if (
                    epochs_without_improvement
                    >= PATIENCE
                ):

                    logger.info(
                        "Early stopping triggered."
                    )

                    break

        # ====================================================
        # FINAL RUN METRICS
        # ====================================================

        mlflow.log_metrics(
            {
                "best_validation_accuracy":
                    best_val_accuracy,
                "best_epoch":
                    float(best_epoch),
            }
        )

        mlflow.log_artifact(
            str(BEST_MODEL_PATH),
            artifact_path="model",
        )

        logger.info(
            "Training completed."
        )

        logger.info(
            "Best validation accuracy: %.4f",
            best_val_accuracy,
        )

        logger.info(
            "Best epoch: %d",
            best_epoch,
        )

        logger.info(
            "Best model path: %s",
            BEST_MODEL_PATH,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        train()

    except Exception:

        logger.exception(
            "Training service failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()