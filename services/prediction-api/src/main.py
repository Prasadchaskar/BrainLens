import logging
import os
import sys

from io import BytesIO

import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from .model_loader import (
    get_device,
    get_model,
    get_model_uri,
)
from .preprocessing import preprocess_image
from .schemas import PredictionResponse

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
# CONFIGURATION
# ============================================================

MODEL_NAME = os.getenv(
    "MLFLOW_MODEL_NAME",
    "brainlens-brain-tumor-classifier",
)

MODEL_ALIAS = os.getenv(
    "MLFLOW_MODEL_ALIAS",
    "champion",
)

CLASS_NAMES = {
    0: "glioma",
    1: "meningioma",
    2: "no_tumor",
}


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="BrainLens Prediction API",
    description=(
        "Brain tumor classification API using "
        "an MLflow-managed EfficientNet-B0 model."
    ),
    version="1.0.0",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Load the champion model once when the API starts.
    """

    logger.info(
        "Starting BrainLens Prediction API..."
    )

    logger.info(
        "MLflow model: %s",
        MODEL_NAME,
    )

    logger.info(
        "MLflow alias: %s",
        MODEL_ALIAS,
    )

    logger.info(
        "Loading champion model..."
    )

    get_model()

    logger.info(
        "Champion model loaded successfully."
    )

    logger.info(
        "Prediction API startup completed."
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
)
def health():
    """
    Basic service health endpoint.
    """

    return {
        "status": "healthy",
        "service": "brainlens-prediction-api",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_uri": get_model_uri(),
        "device": str(get_device()),
    }


# ============================================================
# MODEL HEALTH
# ============================================================

@app.get(
    "/health/model",
)
def model_health():
    """
    Confirm that the MLflow model is loaded.
    """

    try:
        model = get_model()

        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_name": MODEL_NAME,
            "model_alias": MODEL_ALIAS,
            "model_uri": get_model_uri(),
            "device": str(get_device()),
        }

    except Exception as exc:

        logger.exception(
            "Model health check failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Model is unavailable: "
                f"{exc}"
            ),
        )


# ============================================================
# PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
):
    """
    Predict the brain tumor class for an uploaded MRI image.
    """

    # --------------------------------------------------------
    # Validate content type
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="File content type is missing.",
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image type. "
                "Use JPEG or PNG."
            ),
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        file_bytes = await file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        image = Image.open(
            BytesIO(file_bytes)
        )

        image.load()

        image = image.convert(
            "RGB"
        )

    except HTTPException:
        raise

    except Exception as exc:

        logger.exception(
            "Failed to read uploaded image."
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid image file: "
                f"{exc}"
            ),
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:

        model = get_model()

        device = get_device()

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        tensor = preprocess_image(
            image=image,
            device=device,
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with torch.no_grad():

            logits = model(
                tensor
            )

            probabilities = F.softmax(
                logits,
                dim=1,
            )

            confidence, predicted_index = (
                torch.max(
                    probabilities,
                    dim=1,
                )
            )

        predicted_index = int(
            predicted_index.item()
        )

        confidence_value = float(
            confidence.item()
        )

        predicted_class = (
            CLASS_NAMES.get(
                predicted_index,
                "unknown",
            )
        )

        logger.info(
            "Prediction completed | "
            "class=%s | confidence=%.4f",
            predicted_class,
            confidence_value,
        )

        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence_value,
            model_name=MODEL_NAME,
            model_alias=MODEL_ALIAS,
            model_uri=get_model_uri(),
        )

    except Exception as exc:

        logger.exception(
            "Prediction failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{exc}"
            ),
        )