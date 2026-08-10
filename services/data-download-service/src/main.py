import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/app")


def configure_dvc() -> None:
    """Configure DVC credentials from environment variables."""

    username = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not username or not token:
        raise RuntimeError(
            "DAGSHUB_USERNAME and DAGSHUB_TOKEN "
            "environment variables are required."
        )

    logger.info("Configuring DVC remote...")

    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "origin",
            "--local",
            "auth",
            "basic",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "origin",
            "--local",
            "user",
            username,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )

    subprocess.run(
        [
            "dvc",
            "remote",
            "modify",
            "origin",
            "--local",
            "password",
            token,
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def download_data() -> None:
    """Download the dataset from the DVC remote."""

    logger.info("Starting dataset download...")
    logger.info("Project root: %s", PROJECT_ROOT)

    configure_dvc()

    subprocess.run(
        ["dvc", "pull"],
        cwd=PROJECT_ROOT,
        check=True,
    )

    logger.info("Dataset download completed successfully.")


def main() -> None:
    try:
        download_data()

    except Exception:
        logger.exception(
            "Data download service failed."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()