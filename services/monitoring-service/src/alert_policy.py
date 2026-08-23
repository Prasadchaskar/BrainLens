import os

# ============================================================
# CONFIGURATION
# ============================================================

PERFORMANCE_F1_DROP_THRESHOLD = float(
    os.getenv(
        "MONITORING_PERFORMANCE_F1_DROP",
        "0.05",
    )
)


# ============================================================
# ALERT POLICY
# ============================================================

def evaluate_alert_policy(
    monitoring_result: dict,
) -> dict:
    """
    Convert the individual monitoring results into one
    operational alert status.

    Possible statuses:

        HEALTHY
        WARNING
        ALERT
        INSUFFICIENT_DATA
    """

    monitors = monitoring_result.get(
        "monitors",
        {},
    )

    data_drift = monitors.get(
        "data_drift",
        {},
    )

    prediction_drift = monitors.get(
        "prediction_drift",
        {},
    )

    performance = monitors.get(
        "performance",
        {},
    )

    # ========================================================
    # 1. ERROR HAS HIGHEST PRIORITY
    # ========================================================

    for monitor_name, monitor in monitors.items():

        if monitor.get("status") == "ERROR":

            return {
                "status": "ALERT",
                "severity": "high",
                "reason": (
                    f"{monitor_name} monitoring failed."
                ),
                "details": {
                    "monitor": monitor_name,
                    "error": monitor.get(
                        "error"
                    ),
                },
            }

    # ========================================================
    # 2. PERFORMANCE DEGRADATION
    # ========================================================

    performance_status = performance.get(
        "status"
    )

    production_performance = performance.get(
        "performance"
    )

    reference_performance = (
        monitoring_result.get(
            "reference_performance"
        )
    )

    if (
        performance_status == "HEALTHY"
        and production_performance is not None
        and reference_performance is not None
    ):

        production_overall = (
            production_performance.get(
                "overall",
                {},
            )
        )

        reference_overall = (
            reference_performance.get(
                "overall",
                {},
            )
        )

        production_f1 = (
            production_overall.get(
                "f1_weighted"
            )
        )

        reference_f1 = (
            reference_overall.get(
                "f1_weighted"
            )
        )

        if (
            production_f1 is not None
            and reference_f1 is not None
        ):

            f1_drop = (
                reference_f1
                - production_f1
            )

            if (
                f1_drop
                >= PERFORMANCE_F1_DROP_THRESHOLD
            ):

                return {
                    "status": "ALERT",
                    "severity": "high",
                    "reason": (
                        "Production weighted F1 has "
                        "degraded beyond the configured "
                        "threshold."
                    ),
                    "details": {
                        "reference_f1": reference_f1,
                        "production_f1": production_f1,
                        "f1_drop": f1_drop,
                        "threshold": (
                            PERFORMANCE_F1_DROP_THRESHOLD
                        ),
                    },
                }

    # ========================================================
    # 3. DRIFT
    # ========================================================

    drift_sources = []

    if (
        data_drift.get("status")
        == "DRIFT_DETECTED"
    ):
        drift_sources.append(
            "data_drift"
        )

    if (
        prediction_drift.get("status")
        == "DRIFT_DETECTED"
    ):
        drift_sources.append(
            "prediction_drift"
        )

    if drift_sources:

        return {
            "status": "WARNING",
            "severity": "medium",
            "reason": (
                "Distribution drift detected."
            ),
            "details": {
                "sources": drift_sources,
            },
        }

    # ========================================================
    # 4. INSUFFICIENT DATA
    # ========================================================

    statuses = [
        data_drift.get("status"),
        prediction_drift.get("status"),
        performance.get("status"),
    ]

    if all(
        status == "INSUFFICIENT_DATA"
        for status in statuses
    ):

        return {
            "status": "INSUFFICIENT_DATA",
            "severity": "none",
            "reason": (
                "There is not enough recent production "
                "data to perform reliable monitoring."
            ),
            "details": {
                "required_samples": max(
                    data_drift.get(
                        "required_samples",
                        0,
                    ),
                    prediction_drift.get(
                        "required_samples",
                        0,
                    ),
                    performance.get(
                        "required_samples",
                        0,
                    ),
                ),
            },
        }

    # ========================================================
    # 5. MIXED / PARTIAL DATA
    # ========================================================

    if any(
        status == "INSUFFICIENT_DATA"
        for status in statuses
    ):
        return {
            "status": "WARNING",
            "severity": "low",
            "reason": (
                "Monitoring completed, but one or more "
                "checks did not have enough data."
            ),
            "details": {
                "monitor_statuses": {
                    "data_drift": data_drift.get(
                        "status"
                    ),
                    "prediction_drift": (
                        prediction_drift.get(
                            "status"
                        )
                    ),
                    "performance": performance.get(
                        "status"
                    ),
                },
            },
        }

    # ========================================================
    # 6. HEALTHY
    # ========================================================

    return {
        "status": "HEALTHY",
        "severity": "none",
        "reason": (
            "All monitoring checks completed "
            "without detected issues."
        ),
        "details": {},
    }