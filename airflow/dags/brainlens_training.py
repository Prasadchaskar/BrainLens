from datetime import datetime
from airflow.sdk import dag, task
from airflow.providers.docker.operators.docker import DockerOperator
import json

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = (
    "/home/prasad/mlops-projects/BrainLens"
)

DATA_RAW_HOST_PATH = (
    f"{PROJECT_ROOT}/data/raw"
)

ARTIFACTS_HOST_PATH = (
    f"{PROJECT_ROOT}/artifacts"
)

TRAINING_RUN_METADATA_FILE = (
    "/opt/airflow/artifacts/training_run.json"
)

# ============================================================
# DAG
# ============================================================
@dag(
    dag_id="brainlens_training",
    schedule=None,
    start_date=datetime(2026,8,1),
    catchup=False,
    tags=[
        "brainlens",
        "training"
    ]
)
def brainlens_training():
    # Download data
    download_data = DockerOperator(
        task_id="download_data",
        image="brainlens-data-download-service:latest",
        command=["python", "src/main.py"],
        working_dir="/app",
        docker_url=("unix://var/run/docker.sock"),
        environment={
            "DAGSHUB_USERNAME": "{{ conn.dagshub_data.login }}",
            "DAGSHUB_TOKEN": "{{ conn.dagshub_data.password }}",
        },
        auto_remove="success",
        force_pull=False,
    )

    # Validate data
    validate_data = DockerOperator(
        task_id="validate_data",

        image="brainlens-data-validation-service:latest",

        command=[
            "python",
            "src/main.py",
        ],

        working_dir="/app",

        docker_url=(
            "unix://var/run/docker.sock"
        ),

        mounts=[
            {
                "source": DATA_RAW_HOST_PATH,
                "target": "/app/data/raw",
                "type": "bind",
            },
        ],

        environment={
            "PROJECT_ROOT": "/app",
        },

        auto_remove="success",

        force_pull=False,
    )

    # 3. TRAIN MODEL
    train_model = DockerOperator(
        task_id="train_model",

        image="brainlens-training-service:latest",

        command=[
            "python",
            "src/main.py",
        ],

        working_dir="/app",

        docker_url=(
            "unix://var/run/docker.sock"
        ),

        mounts=[
            {
                "source": DATA_RAW_HOST_PATH,
                "target": "/app/data/raw",
                "type": "bind",
            },
            {
                "source": ARTIFACTS_HOST_PATH,
                "target": "/app/artifacts",
                "type": "bind",
            },
        ],

        environment={
            "PROJECT_ROOT": "/app",
            "MLFLOW_TRACKING_URI": (
                "{{ conn.dagshub_mlflow.extra_dejson.get('tracking_uri') }}"
            ),
            "DAGSHUB_USERNAME": (
                "{{ conn.dagshub_data.login }}"
            ),
            "DAGSHUB_TOKEN": (
                "{{ conn.dagshub_data.password }}"
            ),
        },

        auto_remove="success",

        force_pull=False,

        device_requests=[
            {
                "driver": "nvidia",
                "count": -1,
                "capabilities": [
                    ["gpu"],
                ],
            },
        ],
    )

    # 4. READ TRAINING RUN ID

    @task
    def get_training_run_id() -> str:

        with open(
            TRAINING_RUN_METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        run_id = metadata.get(
            "run_id"
        )

        if not run_id:
            raise ValueError(
                "Training run metadata does not "
                "contain run_id."
            )

        print(
            f"Training MLflow run ID: {run_id}"
        )

        return run_id

    # ========================================================
    # 5. EVALUATE MODEL
    # ========================================================

    evaluate_model = DockerOperator(
        task_id="evaluate_model",

        image="brainlens-evaluation-service:latest",

        command=[
            "python",
            "src/main.py",
        ],

        working_dir="/app",

        docker_url=(
            "unix://var/run/docker.sock"
        ),

        mounts=[
            {
                "source": DATA_RAW_HOST_PATH,
                "target": "/app/data/raw",
                "type": "bind",
            },
            {
                "source": ARTIFACTS_HOST_PATH,
                "target": "/app/artifacts",
                "type": "bind",
            },
        ],

        environment={
            "PROJECT_ROOT": "/app",
            "MLFLOW_TRACKING_URI": (
                "{{ conn.dagshub_mlflow.extra_dejson.get('tracking_uri') }}"
            ),
            "DAGSHUB_USERNAME": (
                "{{ conn.dagshub_data.login }}"
            ),
            "DAGSHUB_TOKEN": (
                "{{ conn.dagshub_data.password }}"
            ),
            "TRAINING_RUN_ID": (
                "{{ ti.xcom_pull("
                "task_ids='get_training_run_id'"
                ") }}"
            ),
        },

        auto_remove="success",

        force_pull=False,

        device_requests=[
            {
                "driver": "nvidia",
                "count": -1,
                "capabilities": [
                    ["gpu"],
                ],
            },
        ],
    )

    package_model = DockerOperator(
        task_id="package_model",

        image="brainlens-model-registry-service:latest",

        command=[
            "python",
            "src/package_model.py",
        ],

        working_dir="/app",

        docker_url="unix://var/run/docker.sock",

        mounts=[
            {
                "source": ARTIFACTS_HOST_PATH,
                "target": "/app/artifacts",
                "type": "bind",
            },
        ],

        environment={
            "PROJECT_ROOT": "/app",
            "MLFLOW_TRACKING_URI": (
                "{{ conn.dagshub_mlflow.extra_dejson.get('tracking_uri') }}"
            ),
            "DAGSHUB_USERNAME": (
                "{{ conn.dagshub_data.login }}"
            ),
            "DAGSHUB_TOKEN": (
                "{{ conn.dagshub_data.password }}"
            ),
            "TRAINING_RUN_ID": (
                "{{ ti.xcom_pull("
                "task_ids='get_training_run_id'"
                ") }}"
            ),
        },

        auto_remove="success",

        force_pull=False,

        device_requests=[
            {
                "driver": "nvidia",
                "count": -1,
                "capabilities": [
                    ["gpu"],
                ],
            },
        ],
    )

    register_model = DockerOperator(
        task_id="register_model",

        image="brainlens-model-registry-service:latest",

        command=[
            "python",
            "src/register_model.py",
        ],

        working_dir="/app",

        docker_url="unix://var/run/docker.sock",

        mounts=[
            {
                "source": ARTIFACTS_HOST_PATH,
                "target": "/app/artifacts",
                "type": "bind",
            },
        ],

        environment={
            "PROJECT_ROOT": "/app",
            "MLFLOW_TRACKING_URI": (
                "{{ conn.dagshub_mlflow.extra_dejson.get('tracking_uri') }}"
            ),
            "DAGSHUB_USERNAME": (
                "{{ conn.dagshub_data.login }}"
            ),
            "DAGSHUB_TOKEN": (
                "{{ conn.dagshub_data.password }}"
            ),
        },

        auto_remove="success",

        force_pull=False,
    )
    
    # ========================================================
    # DEPENDENCIES
    # ========================================================

    run_id = get_training_run_id()

    download_data >> validate_data >> train_model

    train_model >> run_id >> evaluate_model

    evaluate_model >> package_model >> register_model
# DAG Instance
brainlens_training()
