import json
from datetime import datetime, timezone
from pathlib import Path
from . import drift
from . import prediction_drift
from . import performance
import os
from .window import (
    DEFAULT_WINDOW_HOURS,
    get_current_window,
)
from . import alert_policy

WINDOW_HOURS = int(
    os.getenv(
        "MONITORING_WINDOW_HOURS",
        str(DEFAULT_WINDOW_HOURS),
    )
)

# ============================================================
# PATHS
# ============================================================

SERVICE_ROOT = (
    Path(__file__).resolve().parents[1]
)

REPORT_DIR = (
    SERVICE_ROOT
    / "data"
    / "reports"
)

MONITORING_RESULT_FILE = (
    REPORT_DIR
    / "monitoring_result.json"
)


# ============================================================
# STATUS PRIORITY
# ============================================================

STATUS_PRIORITY = {
    "ERROR": 4,
    "DRIFT_DETECTED": 3,
    "INSUFFICIENT_DATA": 2,
    "HEALTHY": 1,
}


# ============================================================
# DETERMINE OVERALL STATUS
# ============================================================

def determine_overall_status(
    results: dict,
) -> str:
    """
    Determine the overall monitoring status from
    the three individual monitoring results.

    Priority:

        ERROR
        DRIFT_DETECTED
        INSUFFICIENT_DATA
        HEALTHY
    """

    highest_status = "HEALTHY"

    for result in results.values():

        status = result.get(
            "status",
            "ERROR",
        )

        if (
            STATUS_PRIORITY.get(
                status,
                STATUS_PRIORITY["ERROR"],
            )
            > STATUS_PRIORITY[
                highest_status
            ]
        ):
            highest_status = status

    return highest_status


# ============================================================
# RUN ALL MONITORS
# ============================================================

def run_all_monitors() -> dict:
    """
    Execute all BrainLens monitors using exactly
    the same production monitoring window.
    """

    timestamp = datetime.now(
        timezone.utc
    )

    start_time, end_time = (
        get_current_window(
            WINDOW_HOURS
        )
    )

    results = {}

    # --------------------------------------------------------
    # Data drift
    # --------------------------------------------------------

    try:
        results["data_drift"] = (
            drift.run_monitoring(
                start_time,
                end_time,
            )
        )

    except Exception as exc:

        results["data_drift"] = {
            "status": "ERROR",
            "error": str(exc),
            "window_start": (
                start_time.isoformat()
            ),
            "window_end": (
                end_time.isoformat()
            ),
        }

    # --------------------------------------------------------
    # Prediction drift
    # --------------------------------------------------------

    try:
        results["prediction_drift"] = (
            prediction_drift.run_monitoring(
                start_time,
                end_time,
            )
        )

    except Exception as exc:

        results["prediction_drift"] = {
            "status": "ERROR",
            "error": str(exc),
            "window_start": (
                start_time.isoformat()
            ),
            "window_end": (
                end_time.isoformat()
            ),
        }

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    try:
        results["performance"] = (
            performance.run_monitoring(
                start_time,
                end_time,
            )
        )

    except Exception as exc:

        results["performance"] = {
            "status": "ERROR",
            "error": str(exc),
            "window_start": (
                start_time.isoformat()
            ),
            "window_end": (
                end_time.isoformat()
            ),
        }

    # --------------------------------------------------------
    # Reference performance baseline
    # --------------------------------------------------------

    try:
        reference_performance = (
            performance.calculate_reference_performance()
        )

    except Exception as exc:

        reference_performance = None

        results[
            "reference_performance_error"
        ] = str(exc)

    # --------------------------------------------------------
    # Consolidated result
    # --------------------------------------------------------

    monitoring_result = {
        "monitoring_timestamp": (
            timestamp.isoformat()
        ),
        "window_start": (
            start_time.isoformat()
        ),
        "window_end": (
            end_time.isoformat()
        ),
        "window_hours": WINDOW_HOURS,
        "reference_performance": (
            reference_performance
        ),
        "monitors": results,
    }

    # --------------------------------------------------------
    # Alert policy
    # --------------------------------------------------------

    alert_result = (
        alert_policy.evaluate_alert_policy(
            monitoring_result
        )
    )

    monitoring_result[
        "alert"
    ] = alert_result

    monitoring_result[
        "overall_status"
    ] = alert_result["status"]

    return monitoring_result

# ============================================================
# SAVE RESULT
# ============================================================

# ============================================================
# SAVE MONITORING RESULT
# ============================================================

def save_monitoring_result(
    result: dict,
) -> None:
    """
    Save the latest monitoring result and also preserve
    a historical copy for every monitoring run.
    """

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Latest result
    # --------------------------------------------------------

    latest_file = (
        REPORT_DIR
        / "monitoring_result_latest.json"
    )

    with open(
        latest_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Historical result
    # --------------------------------------------------------

    history_dir = (
        REPORT_DIR
        / "history"
    )

    history_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = result.get(
        "monitoring_timestamp"
    )

    if not timestamp:
        raise ValueError(
            "Monitoring result does not contain "
            "monitoring_timestamp."
        )

    # Convert:
    #
    # 2026-08-23T09:53:38.411302+00:00
    #
    # into a filesystem-friendly name.

    history_name = (
        timestamp
        .replace(
            ":",
            "-",
        )
        .replace(
            "+00:00",
            "",
        )
    )

    history_file = (
        history_dir
        / f"{history_name}.json"
    )

    with open(
        history_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print(
        "\nLatest monitoring result:"
    )

    print(
        latest_file
    )

    print(
        "Historical monitoring result:"
    )

    print(
        history_file
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete BrainLens monitoring pipeline.
    """

    print(
        "Running BrainLens monitoring pipeline..."
    )

    result = run_all_monitors()

    print(
        "\nOverall monitoring status:"
    )

    print(
        result["overall_status"]
    )

    print(
        "\nMonitoring result:"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    save_monitoring_result(
        result
    )

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()