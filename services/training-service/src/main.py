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
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import json


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOGGING
# ============================================================

class TqdmLoggingHandler(logging.Handler):
    """
    Send logger output through tqdm so log messages do not
    corrupt the progress bars.
    """

    def emit(self, record):
        try:
            message = self.format(record)
            tqdm.write(message)
        except Exception:
            self.handleError(record)


sys.stdout.reconfigure(
    line_buffering=True
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        TqdmLoggingHandler()
    ],
    force=True,
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT_ENV = os.getenv(
    "PROJECT_ROOT"
)

if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(
        PROJECT_ROOT_ENV
    )
else:
    PROJECT_ROOT = (
        Path(__file__).resolve().parents[3]
    )


# ============================================================
# DATASET PATHS
# ============================================================
#
# IMPORTANT:
# Use the ORIGINAL DVC dataset.
#
# The research notebook trains directly from:
#
#     ./dataset/train
#
# ResizeWithPadding is performed inside the transforms.
#
# ============================================================

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset"
)

TRAIN_DIR = (
    DATASET_DIR
    / "train"
)


# ============================================================
# MODEL / ARTIFACT CONFIGURATION
# ============================================================

MODEL_NAME = (
    "brainlens-brain-tumor-classifier"
)

BEST_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "best_efficientnet_b0.pth"
)

TRAINING_RUN_METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "training_run.json"
)

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

NUM_CLASSES = 3

NUM_EPOCHS = 1
PATIENCE = 3

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
WEIGHT_DECAY = 1e-4


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
# SUPPORTED IMAGE EXTENSIONS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
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


# ============================================================
# MLFLOW AUTHENTICATION
# ============================================================

os.environ[
    "MLFLOW_TRACKING_USERNAME"
] = DAGSHUB_USERNAME

os.environ[
    "MLFLOW_TRACKING_PASSWORD"
] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# MLFLOW EXPERIMENT
# ============================================================

def get_or_create_experiment(
    experiment_name: str,
) -> str:
    """
    Return an active MLflow experiment ID.
    """

    logger.info(
        "Checking MLflow experiment: %s",
        experiment_name,
    )

    experiment = (
        mlflow.get_experiment_by_name(
            experiment_name
        )
    )

    if experiment is None:

        logger.info(
            "Experiment does not exist. "
            "Creating: %s",
            experiment_name,
        )

        experiment_id = (
            mlflow.create_experiment(
                experiment_name
            )
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

        client = (
            mlflow.tracking.MlflowClient()
        )

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

random.seed(
    SEED
)

np.random.seed(
    SEED
)

torch.manual_seed(
    SEED
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        SEED
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

if torch.cuda.is_available():

    logger.info(
        "GPU: %s",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# DATASET
# ============================================================

class BrainTumorDataset(
    Dataset
):
    """
    Brain tumor image dataset.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
    ):
        self.dataframe = (
            dataframe
            .reset_index(
                drop=True
            )
        )

        self.transform = transform

    def __len__(self):
        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index,
    ):

        row = (
            self.dataframe.iloc[
                index
            ]
        )

        image_path = row[
            "path"
        ]

        class_name = row[
            "class"
        ]

        image = (
            Image.open(
                image_path
            )
            .convert("RGB")
        )

        label = torch.tensor(
            CLASS_TO_INDEX[
                class_name
            ],
            dtype=torch.long,
        )

        if self.transform is not None:
            image = (
                self.transform(
                    image
                )
            )

        return image, label


# ============================================================
# BUILD TRAINING DATAFRAME
# ============================================================

def create_dataframe() -> pd.DataFrame:
    """
    Build the training dataframe exactly according to
    the research notebook.

    No deduplication is performed here.

    Expected:
        3,543 original training images.
    """

    if not TRAIN_DIR.exists():

        raise FileNotFoundError(
            "Training directory does not exist: "
            f"{TRAIN_DIR}"
        )

    records = []

    for class_name in sorted(
        CLASS_TO_INDEX.keys()
    ):

        class_dir = (
            TRAIN_DIR
            / class_name
        )

        if not class_dir.exists():

            raise FileNotFoundError(
                "Missing class directory: "
                f"{class_dir}"
            )

        for image_path in (
            class_dir.rglob("*")
        ):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):

                records.append(
                    {
                        "path": str(
                            image_path
                        ),
                        "class":
                            class_name,
                    }
                )

    dataframe = pd.DataFrame(
        records
    )

    if dataframe.empty:

        raise RuntimeError(
            "No training images found."
        )

    logger.info(
        "Total original training images: %d",
        len(dataframe),
    )

    logger.info(
        "Training distribution:\n%s",
        (
            dataframe["class"]
            .value_counts()
            .sort_index()
            .to_string()
        ),
    )

    return dataframe


# ============================================================
# RESIZE WITH PADDING
# ============================================================

class ResizeWithPadding:
    """
    Preserve aspect ratio and pad to a square.

    Matches the research notebook:
        ResizeWithPadding(224)
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

        width, height = (
            image.size
        )

        if (
            width <= 0
            or height <= 0
        ):

            raise ValueError(
                "Image dimensions must be "
                "greater than zero."
            )

        scale = min(
            self.size / width,
            self.size / height,
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            Image.Resampling.BILINEAR,
        )

        pad_left = (
            self.size
            - new_width
        ) // 2

        pad_top = (
            self.size
            - new_height
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

        return ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ),
            fill=self.fill,
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = (
    transforms.Compose(
        [
            ResizeWithPadding(
                IMAGE_SIZE
            ),

            transforms.RandomAffine(
                degrees=5,
                translate=(
                    0.02,
                    0.02,
                ),
                scale=(
                    0.98,
                    1.02,
                ),
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
)


val_transform = (
    transforms.Compose(
        [
            ResizeWithPadding(
                IMAGE_SIZE
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
)


# ============================================================
# MODEL
# ============================================================

def create_model() -> nn.Module:
    """
    Create pretrained EfficientNet-B0 classifier.
    """

    model = (
        models.efficientnet_b0(
            weights=(
                models
                .EfficientNet_B0_Weights
                .DEFAULT
            )
        )
    )

    logger.info(
        "Loaded pretrained EfficientNet-B0 weights."
    )

    num_features = (
        model
        .classifier[1]
        .in_features
    )

    model.classifier = (
        nn.Sequential(
            nn.Dropout(
                p=0.30
            ),

            nn.Linear(
                num_features,
                NUM_CLASSES,
            ),
        )
    )

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad = (
            True
        )

    model = model.to(
        device
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train() -> None:

    logger.info(
        "Starting training service..."
    )

    # --------------------------------------------------------
    # MLflow experiment
    # --------------------------------------------------------

    experiment_id = (
        get_or_create_experiment(
            MLFLOW_EXPERIMENT_NAME
        )
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    dataframe = create_dataframe()

    # --------------------------------------------------------
    # EXACT NOTEBOOK SPLIT
    # --------------------------------------------------------

    train_df, val_df = (
        train_test_split(
            dataframe,
            test_size=0.20,
            random_state=SEED,
            stratify=dataframe[
                "class"
            ],
        )
    )

    train_df = (
        train_df
        .reset_index(
            drop=True
        )
    )

    val_df = (
        val_df
        .reset_index(
            drop=True
        )
    )

    logger.info(
        "Training dataframe size: %d",
        len(dataframe),
    )

    logger.info(
        "Training images: %d",
        len(train_df),
    )

    logger.info(
        "Validation images: %d",
        len(val_df),
    )

    logger.info(
        "Training distribution:\n%s",
        (
            train_df["class"]
            .value_counts()
            .sort_index()
            .to_string()
        ),
    )

    logger.info(
        "Validation distribution:\n%s",
        (
            val_df["class"]
            .value_counts()
            .sort_index()
            .to_string()
        ),
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = (
        BrainTumorDataset(
            train_df,
            transform=train_transform,
        )
    )

    val_dataset = (
        BrainTumorDataset(
            val_df,
            transform=val_transform,
        )
    )

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            torch.cuda.is_available()
        ),
    )

    logger.info(
        "Training batches: %d",
        len(train_loader),
    )

    logger.info(
        "Validation batches: %d",
        len(val_loader),
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    loss_function = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    model.features.parameters(),
                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    model.classifier.parameters(),
                "lr":
                    CLASSIFIER_LR,
            },
        ],
        weight_decay=(
            WEIGHT_DECAY
        ),
    )

    # --------------------------------------------------------
    # ARTIFACT DIRECTORY
    # --------------------------------------------------------

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
        run_name=(
            "brainlens-efficientnet-b0"
        ),
    ) as run:
        training_run_id = run.info.run_id

        logger.info(
            "MLflow training run ID: %s",
            training_run_id,
        )

        TRAINING_RUN_METADATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            TRAINING_RUN_METADATA_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "run_id": training_run_id,
                    "model_name": MODEL_NAME,
                },
                file,
                indent=2,
            )

        logger.info(
            "Training run metadata saved: %s",
            TRAINING_RUN_METADATA_PATH,
        )

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        mlflow.log_params(
            {
                "model":
                    MODEL_NAME,

                "architecture":
                    "EfficientNet-B0",

                "pretrained":
                    True,

                "image_size":
                    IMAGE_SIZE,

                "resize_strategy":
                    "resize_with_padding",

                "batch_size":
                    BATCH_SIZE,

                "num_classes":
                    NUM_CLASSES,

                "num_epochs":
                    NUM_EPOCHS,

                "patience":
                    PATIENCE,

                "backbone_lr":
                    BACKBONE_LR,

                "classifier_lr":
                    CLASSIFIER_LR,

                "weight_decay":
                    WEIGHT_DECAY,

                "optimizer":
                    "AdamW",

                "loss":
                    "CrossEntropyLoss",

                "seed":
                    SEED,

                "original_train_samples":
                    len(dataframe),

                "train_samples":
                    len(train_dataset),

                "validation_samples":
                    len(val_dataset),

                "device":
                    str(device),
            }
        )

        # ----------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------

        for epoch in range(
            NUM_EPOCHS
        ):

            # =================================================
            # TRAIN
            # =================================================

            model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            train_progress = (
                tqdm(
                    train_loader,
                    desc=(
                        f"Epoch "
                        f"{epoch + 1}/"
                        f"{NUM_EPOCHS} "
                        "Training"
                    ),
                    unit="batch",
                    dynamic_ncols=True,
                )
            )

            for (
                images,
                labels,
            ) in train_progress:

                images = images.to(
                    device,
                    non_blocking=True,
                )

                labels = labels.to(
                    device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                outputs = model(
                    images
                )

                loss = (
                    loss_function(
                        outputs,
                        labels,
                    )
                )

                loss.backward()

                optimizer.step()

                train_loss += (
                    loss.item()
                )

                predictions = (
                    torch.argmax(
                        outputs,
                        dim=1,
                    )
                )

                train_correct += (
                    (
                        predictions
                        == labels
                    )
                    .sum()
                    .item()
                )

                train_total += (
                    labels.size(0)
                )

                batches_done = max(
                    1,
                    train_progress.n,
                )

                current_loss = (
                    train_loss
                    / batches_done
                )

                current_accuracy = (
                    train_correct
                    / max(
                        1,
                        train_total,
                    )
                )

                train_progress.set_postfix(
                    loss=(
                        f"{current_loss:.4f}"
                    ),
                    acc=(
                        f"{current_accuracy:.4f}"
                    ),
                )

            train_loss /= (
                len(train_loader)
            )

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

            val_progress = (
                tqdm(
                    val_loader,
                    desc=(
                        f"Epoch "
                        f"{epoch + 1}/"
                        f"{NUM_EPOCHS} "
                        "Validation"
                    ),
                    unit="batch",
                    dynamic_ncols=True,
                )
            )

            with torch.no_grad():

                for (
                    images,
                    labels,
                ) in val_progress:

                    images = images.to(
                        device,
                        non_blocking=True,
                    )

                    labels = labels.to(
                        device,
                        non_blocking=True,
                    )

                    outputs = model(
                        images
                    )

                    loss = (
                        loss_function(
                            outputs,
                            labels,
                        )
                    )

                    val_loss += (
                        loss.item()
                    )

                    predictions = (
                        torch.argmax(
                            outputs,
                            dim=1,
                        )
                    )

                    val_correct += (
                        (
                            predictions
                            == labels
                        )
                        .sum()
                        .item()
                    )

                    val_total += (
                        labels.size(0)
                    )

                    batches_done = max(
                        1,
                        val_progress.n,
                    )

                    current_loss = (
                        val_loss
                        / batches_done
                    )

                    current_accuracy = (
                        val_correct
                        / max(
                            1,
                            val_total,
                        )
                    )

                    val_progress.set_postfix(
                        loss=(
                            f"{current_loss:.4f}"
                        ),
                        acc=(
                            f"{current_accuracy:.4f}"
                        ),
                    )

            val_loss /= (
                len(val_loader)
            )

            val_accuracy = (
                val_correct
                / val_total
            )

            # =================================================
            # EPOCH SUMMARY
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
            # MLFLOW METRICS
            # =================================================

            mlflow.log_metrics(
                {
                    "train_loss":
                        train_loss,

                    "train_accuracy":
                        train_accuracy,

                    "val_loss":
                        val_loss,

                    "val_accuracy":
                        val_accuracy,
                },
                step=epoch + 1,
            )

            # =================================================
            # BEST MODEL
            # =================================================

            if (
                val_accuracy
                > best_val_accuracy
            ):

                best_val_accuracy = (
                    val_accuracy
                )

                best_epoch = (
                    epoch + 1
                )

                epochs_without_improvement = (
                    0
                )

                torch.save(
                    model.state_dict(),
                    BEST_MODEL_PATH,
                )

                logger.info(
                    "New best model saved. "
                    "Validation accuracy: %.4f",
                    val_accuracy,
                )

                mlflow.log_metric(
                    "best_val_accuracy",
                    best_val_accuracy,
                    step=best_epoch,
                )

            else:

                epochs_without_improvement += (
                    1
                )

                logger.info(
                    "No improvement for "
                    "%d epoch(s). "
                    "Best validation accuracy: "
                    "%.4f",
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
        # FINAL METRICS
        # ====================================================

        mlflow.log_metrics(
            {
                "best_validation_accuracy":
                    best_val_accuracy,

                "best_epoch":
                    float(best_epoch),
            }
        )

        # ====================================================
        # LOG CHECKPOINT ONLY
        # ====================================================
        #
        # IMPORTANT:
        # Training service does NOT call
        # mlflow.pytorch.log_model().
        #
        # Model packaging and registration are handled
        # separately by model-registry-service.
        # ====================================================

        mlflow.log_artifact(
            str(BEST_MODEL_PATH),
            artifact_path="checkpoint",
        )

        # ====================================================
        # TAGS
        # ====================================================

        mlflow.set_tag(
            "model_name",
            MODEL_NAME,
        )

        mlflow.set_tag(
            "model_architecture",
            "EfficientNet-B0",
        )

        mlflow.set_tag(
            "model_stage",
            "candidate",
        )

        mlflow.set_tag(
            "best_epoch",
            str(best_epoch),
        )

        mlflow.set_tag(
            "resize_strategy",
            "resize_with_padding",
        )

        mlflow.set_tag(
            "training_dataset",
            "dvc_raw_dataset",
        )

        logger.info(
            "Training MLflow run completed."
        )

    # ========================================================
    # FINAL LOGGING
    # ========================================================

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
        "Best model checkpoint: %s",
        BEST_MODEL_PATH,
    )

    logger.info(
        "MLflow model name: %s",
        MODEL_NAME,
    )

    logger.info(
        "Model packaging is handled separately "
        "by model-registry-service."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    try:

        train()

    except Exception:

        logger.exception(
            "Training service failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()