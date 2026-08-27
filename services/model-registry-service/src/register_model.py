import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient


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

PROJECT_ROOT_ENV = os.getenv(
    "PROJECT_ROOT"
)

if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(
        PROJECT_ROOT_ENV
    )
else:
    PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI"
)

DAGSHUB_USERNAME = os.getenv(
    "DAGSHUB_USERNAME"
)

DAGSHUB_TOKEN = os.getenv(
    "DAGSHUB_TOKEN"
)

MODEL_NAME = (
    "brainlens-brain-tumor-classifier"
)

PACKAGING_METADATA_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "model_packaging.json"
)


# ============================================================
# VALIDATION
# ============================================================

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

if not PACKAGING_METADATA_PATH.exists():
    raise FileNotFoundError(
        "Model packaging metadata not found: "
        f"{PACKAGING_METADATA_PATH}"
    )


# ============================================================
# MLFLOW AUTH
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
# REGISTER
# ============================================================

def register_model() -> None:
    """
    Register the MLflow LoggedModel produced by
    package_model.py into the MLflow Model Registry.
    """

    # --------------------------------------------------------
    # Load packaging metadata
    # --------------------------------------------------------

    with open(
        PACKAGING_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    logged_model_id = metadata.get(
        "logged_model_id"
    )

    if not logged_model_id:
        raise ValueError(
            "model_packaging.json does not contain "
            "logged_model_id."
        )

    logger.info(
        "Logged model ID: %s",
        logged_model_id,
    )

    # --------------------------------------------------------
    # MLflow client
    # --------------------------------------------------------

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    # --------------------------------------------------------
    # Fetch logged model
    # --------------------------------------------------------

    logged_model = (
        client.get_logged_model(
            logged_model_id
        )
    )

    logger.info(
        "Logged model found."
    )

    logger.info(
        "Logged model name: %s",
        logged_model.name,
    )

    logger.info(
        "Source run: %s",
        logged_model.source_run_id,
    )

    # MLflow exposes the logged model URI.
    model_uri = logged_model.model_uri

    logger.info(
        "Model URI: %s",
        model_uri,
    )

    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------

    logger.info(
        "Registering as: %s",
        MODEL_NAME,
    )

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME,
    )

    logger.info(
        "Model registered successfully."
    )

    logger.info(
        "Registered model: %s",
        model_version.name,
    )

    logger.info(
        "Registered version: %s",
        model_version.version,
    )

    # --------------------------------------------------------
    # Version metadata
    # --------------------------------------------------------

    client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="model_stage",
        value="candidate",
    )

    client.set_model_version_tag(
        name=model_version.name,
        version=model_version.version,
        key="source_logged_model_id",
        value=logged_model_id,
    )

    if metadata.get(
        "training_run_id"
    ):
        client.set_model_version_tag(
            name=model_version.name,
            version=model_version.version,
            key="training_run_id",
            value=metadata[
                "training_run_id"
            ],
        )

    logger.info(
        "Model version tagged as candidate."
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    try:

        register_model()

    except Exception:

        logger.exception(
            "Model registration failed."
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()