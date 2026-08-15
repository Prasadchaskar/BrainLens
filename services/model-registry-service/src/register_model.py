import logging
import os
import sys

import mlflow

from dotenv import load_dotenv
from mlflow import MlflowClient


load_dotenv()


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)


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

# This is the Logged Model ID shown in your MLflow UI.
LOGGED_MODEL_ID = (
    "m-e79c42a22076447fba6463b742c790ff"
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


# ============================================================
# MLFLOW AUTH
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
# REGISTER
# ============================================================

def register_model() -> None:

    client = MlflowClient(
        tracking_uri=MLFLOW_TRACKING_URI
    )

    logger.info(
        "Fetching logged model: %s",
        LOGGED_MODEL_ID,
    )

    logged_model = client.get_logged_model(
        LOGGED_MODEL_ID
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

    # MLflow exposes the logged model's model URI.
    model_uri = logged_model.model_uri

    logger.info(
        "Model URI: %s",
        model_uri,
    )

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
        value=LOGGED_MODEL_ID,
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


if __name__ == "__main__":
    main()