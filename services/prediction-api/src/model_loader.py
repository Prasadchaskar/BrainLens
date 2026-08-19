import logging
import os
from typing import Optional

import mlflow
import mlflow.pytorch
import torch

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)

MLFLOW_USERNAME = os.getenv(
    "DAGSHUB_USERNAME"
)

MLFLOW_TOKEN = os.getenv(
    "DAGSHUB_TOKEN"
)

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "brainlens-brain-tumor-classifier",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)


# ============================================================
# VALIDATION
# ============================================================

if not MLFLOW_TRACKING_URI:
    raise RuntimeError(
        "MLFLOW_TRACKING_URI is required."
    )

if not MLFLOW_USERNAME:
    raise RuntimeError(
        "DAGSHUB_USERNAME is required."
    )

if not MLFLOW_TOKEN:
    raise RuntimeError(
        "DAGSHUB_TOKEN is required."
    )


# ============================================================
# MLFLOW AUTHENTICATION
# ============================================================

os.environ[
    "MLFLOW_TRACKING_USERNAME"
] = MLFLOW_USERNAME

os.environ[
    "MLFLOW_TRACKING_PASSWORD"
] = MLFLOW_TOKEN


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL STATE
# ============================================================

_model = None
_model_uri: Optional[str] = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model():
    """
    Load the model assigned to the configured MLflow alias.

    Example:
        models:/brainlens-brain-tumor-classifier@champion
    """

    global _model
    global _model_uri

    if _model is not None:
        return _model

    _model_uri = (
        f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    )

    logger.info(
        "Loading MLflow model: %s",
        _model_uri,
    )

    logger.info(
        "Using device: %s",
        DEVICE,
    )

    model = mlflow.pytorch.load_model(
        _model_uri,
        map_location=DEVICE,
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    _model = model

    logger.info(
        "Model loaded successfully."
    )

    return _model


# ============================================================
# MODEL ACCESS
# ============================================================

def get_model():
    """
    Return the cached model.
    """

    if _model is None:
        return load_model()

    return _model


def get_model_uri() -> str:
    """
    Return the configured MLflow model URI.
    """

    if _model_uri is None:
        return (
            f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )

    return _model_uri


def get_device() -> torch.device:
    """
    Return the inference device.
    """

    return DEVICE