import logging
import os
import sys
from pathlib import Path
import json
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from dotenv import load_dotenv
from mlflow.models import infer_signature


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
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True,
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

MODEL_PACKAGING_METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "model_packaging.json"
)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = (
    "brainlens-brain-tumor-classifier"
)

MODEL_ARTIFACT_NAME = "model"

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "best_efficientnet_b0.pth"
)

IMAGE_SIZE = 224
NUM_CLASSES = 3

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "MLFLOW_EXPERIMENT_NAME",
    "brainlens-training",
)

TRAINING_RUN_ID = os.getenv(
    "TRAINING_RUN_ID"
)

DAGSHUB_USERNAME = os.getenv(
    "DAGSHUB_USERNAME"
)

DAGSHUB_TOKEN = os.getenv(
    "DAGSHUB_TOKEN"
)


# ============================================================
# VALIDATION
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
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

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for PT2 packaging in this service. "
        "Run the packaging service on the GPU environment."
    )


# ============================================================
# MLFLOW AUTHENTICATION
# ============================================================

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
# MODEL ARCHITECTURE
# ============================================================

def create_model() -> nn.Module:
    """
    Recreate the exact EfficientNet-B0 architecture
    used by the training service.
    """

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


# ============================================================
# GET EXPERIMENT
# ============================================================

def get_experiment():
    """Get the active MLflow experiment."""

    experiment = (
        mlflow.get_experiment_by_name(
            MLFLOW_EXPERIMENT_NAME
        )
    )

    if experiment is None:
        raise RuntimeError(
            "MLflow experiment does not exist: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    if experiment.lifecycle_stage == "deleted":
        raise RuntimeError(
            "MLflow experiment is deleted: "
            f"{MLFLOW_EXPERIMENT_NAME}"
        )

    logger.info(
        "Using MLflow experiment: %s",
        experiment.experiment_id,
    )

    return experiment


# ============================================================
# PACKAGE MODEL
# ============================================================

def package_model() -> None:
    """
    Package the existing best checkpoint into a real
    MLflow PyTorch model.

    NO TRAINING happens here.
    """

    logger.info(
        "Starting model packaging..."
    )

    logger.info(
        "Checkpoint: %s",
        MODEL_PATH,
    )

    logger.info(
        "Model name: %s",
        MODEL_NAME,
    )

    logger.info(
        "Using CUDA device: %s",
        torch.cuda.get_device_name(0),
    )

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # LOAD EXISTING BEST CHECKPOINT
    # --------------------------------------------------------

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cuda",
    )

    model.load_state_dict(
        state_dict
    )

    model = model.cuda()
    model.eval()

    logger.info(
        "Best checkpoint loaded on CUDA."
    )

    # --------------------------------------------------------
    # INPUT EXAMPLE
    #
    # IMPORTANT:
    # Model and example are BOTH on CUDA for PT2 export.
    # --------------------------------------------------------

    input_example = np.zeros(
        (
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        dtype=np.float32,
    )

    input_tensor = torch.from_numpy(
        input_example
    ).cuda()

    # --------------------------------------------------------
    # OUTPUT EXAMPLE
    # --------------------------------------------------------

    with torch.no_grad():

        output_tensor = model(
            input_tensor
        )

    output_example = (
        output_tensor
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # SIGNATURE
    #
    # Signature uses CPU NumPy arrays for the model schema.
    # The actual PT2 export still uses CUDA tensors.
    # --------------------------------------------------------

    signature = infer_signature(
        input_example,
        output_example,
    )

    # --------------------------------------------------------
    # EXPERIMENT
    # --------------------------------------------------------

    experiment = get_experiment()

    # --------------------------------------------------------
    # CREATE PACKAGING RUN
    # --------------------------------------------------------

    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name="brainlens-model-packaging-pt2",
    ):

        # ----------------------------------------------------
        # TAGS
        # ----------------------------------------------------

        if TRAINING_RUN_ID:
            mlflow.set_tag(
                "training_run_id",
                TRAINING_RUN_ID,
            )

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
            "packaging_source",
            "best_efficientnet_b0.pth",
        )

        mlflow.set_tag(
            "serialization_format",
            "pt2",
        )

        mlflow.set_tag(
            "device",
            "cuda",
        )

        # ----------------------------------------------------
        # MODEL LOGGING
        # ----------------------------------------------------

        logger.info(
            "Logging PyTorch model using safe PT2 format..."
        )

        model_info = (
            mlflow.pytorch.log_model(
                model,
                name=MODEL_ARTIFACT_NAME,
                signature=signature,
                input_example=input_tensor,
                serialization_format="pt2",
            )
        )

        logger.info(
            "MLflow model logged successfully."
        )

        logger.info(
            "Model URI: %s",
            model_info.model_uri,
        )

        logged_model_id = model_info.model_id

        logger.info(
            "Logged model ID: %s",
            logged_model_id,
        )

        MODEL_PACKAGING_METADATA_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            MODEL_PACKAGING_METADATA_PATH,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "logged_model_id": logged_model_id,
                    "model_uri": model_info.model_uri,
                    "model_name": MODEL_NAME,
                    "training_run_id": TRAINING_RUN_ID,
                },
                file,
                indent=2,
            )

        logger.info(
            "Model packaging metadata saved: %s",
            MODEL_PACKAGING_METADATA_PATH,
        )

        # ----------------------------------------------------
        # CHECKPOINT
        # ----------------------------------------------------

        mlflow.log_artifact(
            str(MODEL_PATH),
            artifact_path="checkpoint",
        )

        logger.info(
            "Checkpoint logged successfully."
        )

    logger.info(
        "Model packaging completed successfully."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    try:
        package_model()

    except Exception:

        logger.exception(
            "Model packaging failed."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()