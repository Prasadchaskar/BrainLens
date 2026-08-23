import logging
import sys

from fastapi import FastAPI, HTTPException

from .schemas import PredictionEvent, GroundTruthUpdate
from .storage import (
    initialize_storage,
    save_prediction_event,
    update_actual_class
)


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
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="BrainLens Monitoring Service",
    description=(
        "Monitoring event ingestion service for "
        "the BrainLens prediction API."
    ),
    version="1.0.0",
)


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    """
    Initialize monitoring storage when the service starts.
    """

    logger.info(
        "Starting BrainLens Monitoring Service..."
    )

    initialize_storage()

    logger.info(
        "Monitoring storage initialized."
    )

    logger.info(
        "BrainLens Monitoring Service startup completed."
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    """
    Basic monitoring-service health endpoint.
    """

    return {
        "status": "healthy",
        "service": "brainlens-monitoring-service",
    }


# ============================================================
# EVENT INGESTION
# ============================================================

@app.post("/events")
def ingest_prediction_event(
    event: PredictionEvent,
):
    """
    Store one prediction monitoring event.
    """

    try:
        save_prediction_event(event)

        logger.info(
            "Prediction event stored | "
            "prediction_id=%s | "
            "model_version=%s | "
            "predicted_class=%s | "
            "confidence=%.4f",
            event.prediction_id,
            event.model_version,
            event.predicted_class,
            event.confidence,
        )

        return {
            "status": "accepted",
            "prediction_id": event.prediction_id,
        }

    except Exception as exc:
        logger.exception(
            "Failed to store prediction event."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to store prediction event: "
                f"{exc}"
            ),
        )
    

# ============================================================
# GROUND-TRUTH INGESTION
# ============================================================

@app.post(
    "/events/{prediction_id}/ground-truth"
)
def update_prediction_ground_truth(
    prediction_id: str,
    ground_truth: GroundTruthUpdate,
):
    """
    Attach a confirmed ground-truth label to an existing
    prediction event.
    """

    try:
        updated = update_actual_class(
            prediction_id=prediction_id,
            actual_class=ground_truth.actual_class,
        )

        if not updated:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Prediction event not found: "
                    f"{prediction_id}"
                ),
            )

        logger.info(
            "Ground truth updated | "
            "prediction_id=%s | "
            "actual_class=%s",
            prediction_id,
            ground_truth.actual_class,
        )

        return {
            "status": "updated",
            "prediction_id": prediction_id,
            "actual_class": ground_truth.actual_class,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed to update ground truth."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to update ground truth: "
                f"{exc}"
            ),
        )