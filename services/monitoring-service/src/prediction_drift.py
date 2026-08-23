import os
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from .storage import load_prediction_events
from .window import (
    DEFAULT_WINDOW_HOURS,
    get_current_window,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DRIFT_THRESHOLD = 0.50

COLUMN_DRIFT_P_VALUE_THRESHOLD = 0.05

WINDOW_HOURS = int(
    os.getenv(
        "MONITORING_WINDOW_HOURS",
        str(DEFAULT_WINDOW_HOURS),
    )
)

MIN_CURRENT_SAMPLES = int(
    os.getenv(
        "MONITORING_MIN_CURRENT_SAMPLES",
        "30",
    )
)


# ============================================================
# PATHS
# ============================================================

SERVICE_ROOT = (
    Path(__file__).resolve().parents[1]
)

REFERENCE_FILE = (
    SERVICE_ROOT
    / "data"
    / "reference"
    / "brainlens_reference_predictions.parquet"
)

REPORT_DIR = (
    SERVICE_ROOT
    / "data"
    / "reports"
)

REPORT_FILE = (
    REPORT_DIR
    / "prediction_drift_report.html"
)


# ============================================================
# MONITORED PREDICTION FEATURES
# ============================================================

PREDICTION_DRIFT_COLUMNS = [
    "predicted_class",
    "confidence",
]


# ============================================================
# LOAD REFERENCE PREDICTIONS
# ============================================================

def load_reference_predictions() -> pd.DataFrame:
    """
    Load the fixed reference prediction dataset.

    These predictions were generated through the real
    BrainLens prediction API.
    """

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            "Reference prediction dataset not found: "
            f"{REFERENCE_FILE}"
        )

    dataframe = pd.read_parquet(
        REFERENCE_FILE
    )

    if dataframe.empty:
        raise ValueError(
            "Reference prediction dataset is empty."
        )

    return dataframe


# ============================================================
# PREPARE PREDICTION DATA
# ============================================================

def prepare_prediction_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select only the model-output columns used for
    prediction-drift monitoring.
    """

    missing_columns = [
        column
        for column in PREDICTION_DRIFT_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing prediction-drift columns: "
            f"{missing_columns}"
        )

    prepared = dataframe[
        PREDICTION_DRIFT_COLUMNS
    ].copy()

    prepared["confidence"] = pd.to_numeric(
        prepared["confidence"],
        errors="coerce",
    )

    if prepared["confidence"].isna().any():
        raise ValueError(
            "Current/reference prediction data "
            "contains invalid confidence values."
        )

    return prepared


# ============================================================
# RUN PREDICTION DRIFT
# ============================================================

def run_prediction_drift(
    current_data: pd.DataFrame,
):
    """
    Compare current production model outputs against
    the fixed reference prediction population.
    """

    reference_data = (
        load_reference_predictions()
    )

    reference_data = (
        prepare_prediction_data(
            reference_data
        )
    )

    current_data = (
        prepare_prediction_data(
            current_data
        )
    )

    report = Report(
        [
            DataDriftPreset(),
        ]
    )

    evaluation = report.run(
        current_data,
        reference_data,
    )

    summary = (
        get_prediction_drift_summary(
            evaluation
        )
    )

    return evaluation, summary


# ============================================================
# GET PREDICTION DRIFT SUMMARY
# ============================================================

def get_prediction_drift_summary(
    evaluation,
) -> dict:
    """
    Convert the Evidently Snapshot into a
    machine-readable BrainLens prediction-drift summary.
    """

    data = evaluation.dict()

    metrics = data.get(
        "metrics",
        [],
    )

    total_columns = len(
        PREDICTION_DRIFT_COLUMNS
    )

    drifted_columns = 0

    drift_share = 0.0

    drifted_features = []

    # --------------------------------------------------------
    # Dataset-level drift
    # --------------------------------------------------------

    for metric in metrics:

        metric_name = metric.get(
            "metric_name",
            "",
        )

        if metric_name.startswith(
            "DriftedColumnsCount"
        ):

            value = metric.get(
                "value",
                {},
            )

            drifted_columns = int(
                value.get(
                    "count",
                    0,
                )
            )

            drift_share = float(
                value.get(
                    "share",
                    0.0,
                )
            )

            break

    # --------------------------------------------------------
    # Feature-level drift
    # --------------------------------------------------------

    for metric in metrics:

        metric_name = metric.get(
            "metric_name",
            "",
        )

        prefix = "ValueDrift(column="

        if not metric_name.startswith(
            prefix
        ):
            continue

        value = metric.get(
            "value"
        )

        if value is None:
            continue

        remainder = metric_name[
            len(prefix):
        ]

        column_name = remainder.split(
            ",",
            1,
        )[0]

        p_value = float(
            value
        )

        if (
            p_value
            < COLUMN_DRIFT_P_VALUE_THRESHOLD
        ):
            drifted_features.append(
                {
                    "feature": column_name,
                    "p_value": p_value,
                }
            )

    return {
        "dataset_drift": (
            drift_share
            >= DATASET_DRIFT_THRESHOLD
        ),
        "drifted_columns": (
            drifted_columns
        ),
        "total_columns": (
            total_columns
        ),
        "drift_share": (
            drift_share
        ),
        "drifted_features": (
            drifted_features
        ),
    }


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    evaluation,
) -> None:
    """
    Save the Evidently prediction-drift report as HTML.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.save_html(
        str(REPORT_FILE)
    )

    if not REPORT_FILE.exists():
        raise RuntimeError(
            "Prediction drift report was not created: "
            f"{REPORT_FILE}"
        )

    report_size = (
        REPORT_FILE.stat().st_size
    )

    if report_size == 0:
        raise RuntimeError(
            "Prediction drift report is empty."
        )

    print(
        "Prediction drift report created successfully:"
    )

    print(
        REPORT_FILE
    )

    print(
        f"Report size: {report_size:,} bytes"
    )

# ============================================================
# MONITORING FUNCTION
# ============================================================

def run_monitoring(start_time, end_time) -> dict:
    """
    Run prediction-drift monitoring for the current
    production monitoring window.

    Returns a structured monitoring result.
    """
    current_data = load_prediction_events(
        start_time=start_time,
        end_time=end_time,
    )

    current_count = len(
        current_data
    )

    # --------------------------------------------------------
    # Minimum sample-size gate
    # --------------------------------------------------------

    if current_count < MIN_CURRENT_SAMPLES:
        return {
            "status": "INSUFFICIENT_DATA",
            "window_start": start_time.isoformat(),
            "window_end": end_time.isoformat(),
            "window_hours": (end_time - start_time).total_seconds() / 3600,
            "sample_count": current_count,
            "required_samples": MIN_CURRENT_SAMPLES,
            "drifted_columns": 0,
            "total_columns": len(
                PREDICTION_DRIFT_COLUMNS
            ),
            "drift_share": 0.0,
            "drifted_features": [],
        }

    # --------------------------------------------------------
    # Run Evidently
    # --------------------------------------------------------

    evaluation, summary = (
        run_prediction_drift(
            current_data
        )
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    save_report(
        evaluation
    )

    return {
        "status": (
            "DRIFT_DETECTED"
            if summary["dataset_drift"]
            else "HEALTHY"
        ),
        "window_start": start_time.isoformat(),
        "window_end": end_time.isoformat(),
        "window_hours": WINDOW_HOURS,
        "sample_count": current_count,
        **summary,
    }

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run prediction-drift monitoring from the command line.
    """

    print(
        "Running BrainLens prediction-drift monitoring..."
    )

    result = run_monitoring()

    print(
        "\nPrediction drift result:"
    )

    print(
        result
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()