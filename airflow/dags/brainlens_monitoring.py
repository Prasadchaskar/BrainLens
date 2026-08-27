import json
from datetime import datetime

from airflow.sdk import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.smtp.operators.smtp import EmailOperator


# ============================================================
# PATHS
# ============================================================

MONITORING_DATA_HOST_PATH = (
    "/home/prasad/mlops-projects/BrainLens/"
    "services/monitoring-service/data"
)

MONITORING_RESULT_FILE = (
    "/opt/airflow/monitoring-data/"
    "reports/monitoring_result_latest.json"
)


# ============================================================
# EMAIL CONFIGURATION
# ============================================================

ALERT_EMAIL_TO = (
    "prasadtest67@gmail.com"
)


# ============================================================
# DAG
# ============================================================

@dag(
    dag_id="brainlens_monitoring",
    schedule=None,
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=[
        "brainlens",
        "monitoring",
    ],
)
def brainlens_monitoring():

    # ========================================================
    # 1. RUN MONITORING
    # ========================================================

    run_monitoring = DockerOperator(
        task_id="run_monitoring",

        image="brainlens-monitoring-service:latest",

        command=[
            "python",
            "-m",
            "src.monitor",
        ],

        working_dir="/app",

        docker_url=(
            "unix://var/run/docker.sock"
        ),

        network_mode="brainlens-network",

        mounts=[
            {
                "source": (
                    MONITORING_DATA_HOST_PATH
                ),
                "target": "/app/data",
                "type": "bind",
            },
        ],

        auto_remove="success",

        force_pull=False,
    )

    # ========================================================
    # 2. READ MONITORING RESULT
    # ========================================================

    @task
    def read_monitoring_result():
        """
        Read the consolidated monitoring result and return
        only the information required by Airflow for
        branching and notifications.
        """

        with open(
            MONITORING_RESULT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            result = json.load(file)

        monitors = result.get(
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

        performance_result = (
            performance.get(
                "performance"
            )
        )

        performance_overall = {}

        if performance_result:
            performance_overall = (
                performance_result.get(
                    "overall",
                    {},
                )
            )

        payload = {
            "overall_status": result.get(
                "overall_status"
            ),
            "alert_status": result.get(
                "alert",
                {},
            ).get(
                "status"
            ),
            "alert_reason": result.get(
                "alert",
                {},
            ).get(
                "reason"
            ),
            "alert_details": result.get(
                "alert",
                {},
            ).get(
                "details",
                {},
            ),
            "monitoring_timestamp": result.get(
                "monitoring_timestamp"
            ),
            "window_start": result.get(
                "window_start"
            ),
            "window_end": result.get(
                "window_end"
            ),
            "window_hours": result.get(
                "window_hours"
            ),
            "data_drift_status": (
                data_drift.get(
                    "status"
                )
            ),
            "data_drifted_features": (
                data_drift.get(
                    "drifted_features",
                    [],
                )
            ),
            "prediction_drift_status": (
                prediction_drift.get(
                    "status"
                )
            ),
            "prediction_drifted_features": (
                prediction_drift.get(
                    "drifted_features",
                    [],
                )
            ),
            "performance_status": (
                performance.get(
                    "status"
                )
            ),
            "production_sample_count": (
                performance.get(
                    "sample_count"
                )
            ),
            "labeled_sample_count": (
                performance.get(
                    "labeled_sample_count"
                )
            ),
            "production_f1": (
                performance_overall.get(
                    "f1_weighted"
                )
            ),
            "reference_f1": (
                result.get(
                    "reference_performance",
                    {},
                )
                .get(
                    "overall",
                    {},
                )
                .get(
                    "f1_weighted"
                )
            ),
        }

        print(
            json.dumps(
                payload,
                indent=2,
            )
        )

        return payload

    # ========================================================
    # 3. CHOOSE ACTION
    # ========================================================

    @task.branch
    def choose_action(
        payload: dict,
    ):
        """
        Select the next task based on the operational
        monitoring status.
        """

        status = payload.get(
            "overall_status"
        )

        if status == "ALERT":
            return "send_alert_email"

        if status == "WARNING":
            return "send_warning_email"

        if status == "INSUFFICIENT_DATA":
            return "handle_insufficient_data"

        if status == "HEALTHY":
            return "handle_healthy"

        raise ValueError(
            f"Unknown monitoring status: {status}"
        )

    # ========================================================
    # 4. HEALTHY
    # ========================================================

    @task
    def handle_healthy():

        print(
            "BrainLens monitoring is healthy. "
            "No action required."
        )

    # ========================================================
    # 5. INSUFFICIENT DATA
    # ========================================================

    @task
    def handle_insufficient_data():

        print(
            "BrainLens monitoring does not have "
            "enough recent production data."
        )

    # ========================================================
    # 6. WARNING EMAIL
    # ========================================================

    send_warning_email = EmailOperator(
        task_id="send_warning_email",

        to=ALERT_EMAIL_TO,

        conn_id="smtp_default",

        subject=(
            "[BrainLens WARNING] "
            "Monitoring issue detected"
        ),

        html_content="""
        <h2>BrainLens Monitoring Warning</h2>

        <p>
            A non-critical monitoring issue was detected.
        </p>

        <p>
            <strong>Status:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['overall_status'] }}
        </p>

        <p>
            <strong>Reason:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['alert_reason'] }}
        </p>

        <p>
            <strong>Monitoring window:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['window_start'] }}
            →
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['window_end'] }}
        </p>

        <h3>Data Drift</h3>

        <p>
            Status:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['data_drift_status'] }}
        </p>

        <p>
            Drifted features:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['data_drifted_features'] }}
        </p>

        <h3>Prediction Drift</h3>

        <p>
            Status:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['prediction_drift_status'] }}
        </p>

        <p>
            Drifted features:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['prediction_drifted_features'] }}
        </p>

        <p>
            Review the BrainLens monitoring reports
            for further investigation.
        </p>
        """,
    )

    # ========================================================
    # 7. ALERT EMAIL
    # ========================================================

    send_alert_email = EmailOperator(
        task_id="send_alert_email",

        to=ALERT_EMAIL_TO,

        conn_id="smtp_default",

        subject=(
            "[BrainLens ALERT] "
            "Production monitoring alert"
        ),

        html_content="""
        <h2>BrainLens Production Alert</h2>

        <p>
            BrainLens monitoring detected a production
            issue requiring attention.
        </p>

        <p>
            <strong>Status:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['overall_status'] }}
        </p>

        <p>
            <strong>Reason:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['alert_reason'] }}
        </p>

        <p>
            <strong>Monitoring window:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['window_start'] }}
            →
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['window_end'] }}
        </p>

        <h3>Performance</h3>

        <p>
            <strong>Reference F1:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['reference_f1'] }}
        </p>

        <p>
            <strong>Production F1:</strong>
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['production_f1'] }}
        </p>

        <h3>Data Drift</h3>

        <p>
            Status:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['data_drift_status'] }}
        </p>

        <p>
            Drifted features:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['data_drifted_features'] }}
        </p>

        <h3>Prediction Drift</h3>

        <p>
            Status:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['prediction_drift_status'] }}
        </p>

        <p>
            Drifted features:
            {{ ti.xcom_pull(
                task_ids='read_monitoring_result'
            )['prediction_drifted_features'] }}
        </p>

        <p>
            <strong>Action required:</strong>
            investigate the production monitoring results.
        </p>
        """,
    )

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    payload = read_monitoring_result()

    action = choose_action(
        payload
    )

    run_monitoring >> payload >> action

    action >> [
        handle_healthy(),
        handle_insufficient_data(),
        send_warning_email,
        send_alert_email,
    ]


# ============================================================
# DAG INSTANCE
# ============================================================

brainlens_monitoring()