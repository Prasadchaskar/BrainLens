from pathlib import Path
import pandas as pd
from .schemas import PredictionEvent
from datetime import datetime, timedelta, timezone
from typing import Optional

# ============================================================
# PATHS
# ============================================================

SERVICE_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = SERVICE_ROOT / "data"

PREDICTIONS_FILE = DATA_DIR / "prediction_events.csv"


# ============================================================
# SCHEMA
# ============================================================

COLUMNS = [
    "prediction_id",
    "timestamp",
    "model_name",
    "model_alias",
    "model_version",
    "predicted_class",
    "confidence",
    "original_image_width",
    "original_image_height",
    "original_aspect_ratio",
    "processed_image_width",
    "processed_image_height",
    "mean_r",
    "mean_g",
    "mean_b",
    "std_r",
    "std_g",
    "std_b",
    "brightness",
    "contrast",
    "actual_class",
]

# ============================================================
# INITIALIZE STORAGE
# ============================================================

def initialize_storage() -> None:
    """
    Create the monitoring data directory and CSV file
    if they do not already exist.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PREDICTIONS_FILE.exists():

        dataframe = pd.DataFrame(
            columns=COLUMNS
        )

        dataframe.to_csv(
            PREDICTIONS_FILE,
            index=False,
        )


# ============================================================
# SAVE PREDICTION EVENT
# ============================================================

def save_prediction_event(
    event: PredictionEvent,
) -> None:
    """
    Append a PredictionEvent to the event store.
    """

    initialize_storage()

    record = event.model_dump()

    dataframe = pd.DataFrame(
        [record],
        columns=COLUMNS,
    )

    dataframe.to_csv(
        PREDICTIONS_FILE,
        mode="a",
        header=False,
        index=False,
    )


# ============================================================
# LOAD PREDICTION EVENTS
# ============================================================

def load_prediction_events(
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Load prediction events, optionally restricted to a
    time window.

    Timestamps are interpreted as UTC.
    """

    initialize_storage()

    dataframe = pd.read_csv(
        PREDICTIONS_FILE
    )

    if dataframe.empty:
        return dataframe

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
        utc=True,
    )

    if start_time is not None:
        start_time = start_time.astimezone(
            timezone.utc
        )

        dataframe = dataframe[
            dataframe["timestamp"] >= start_time
        ]

    if end_time is not None:
        end_time = end_time.astimezone(
            timezone.utc
        )

        dataframe = dataframe[
            dataframe["timestamp"] < end_time
        ]

    return dataframe.reset_index(
        drop=True
    )


# ============================================================
# UPDATE GROUND TRUTH
# ============================================================

def update_actual_class(
    prediction_id: str,
    actual_class: str,
) -> bool:
    """
    Attach the actual class to an existing prediction.
    """

    dataframe = load_prediction_events()

    mask = (
        dataframe["prediction_id"]
        == prediction_id
    )

    if not mask.any():
        return False

    dataframe.loc[
        mask,
        "actual_class",
    ] = actual_class

    dataframe.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    return True