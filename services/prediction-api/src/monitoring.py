import logging
import os

import httpx


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

MONITORING_SERVICE_URL = os.getenv(
    "MONITORING_SERVICE_URL",
    "http://host.docker.internal:8001",
)

MONITORING_EVENT_TIMEOUT = float(
    os.getenv(
        "MONITORING_EVENT_TIMEOUT",
        "2.0",
    )
)


# ============================================================
# SEND PREDICTION EVENT
# ============================================================

def send_prediction_event(
    event: dict,
) -> None:
    """
    Send a prediction monitoring event to the monitoring service.

    Monitoring failures are intentionally non-fatal.
    Prediction inference should continue even if the monitoring
    service is temporarily unavailable.
    """

    endpoint = (
        f"{MONITORING_SERVICE_URL.rstrip('/')}"
        "/events"
    )

    try:
        response = httpx.post(
            endpoint,
            json=event,
            timeout=MONITORING_EVENT_TIMEOUT,
        )

        response.raise_for_status()

        logger.info(
            "Prediction event sent successfully | "
            "prediction_id=%s",
            event.get("prediction_id"),
        )

    except Exception as exc:
        logger.warning(
            "Failed to send prediction event | "
            "prediction_id=%s | "
            "error=%s",
            event.get("prediction_id"),
            exc,
        )