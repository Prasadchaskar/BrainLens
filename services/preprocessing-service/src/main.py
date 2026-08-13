import logging
import os
import sys
from pathlib import Path

from PIL import Image


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")

if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(PROJECT_ROOT_ENV)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]


RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"
PROCESSED_DATASET_DIR = PROJECT_ROOT / "data" / "processed" / "dataset"

IMAGE_SIZE = (224, 224)

EXPECTED_SPLITS = {
    "train",
    "test",
}

EXPECTED_CLASSES = {
    "glioma",
    "meningioma",
    "no_tumor",
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


def preprocess_image(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Resize and convert one image to RGB."""

    with Image.open(source_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            destination_path,
            format="JPEG",
        )


def preprocess_dataset() -> tuple[int, int]:
    """Preprocess the complete dataset."""

    logger.info("Starting dataset preprocessing...")
    logger.info("Input dataset: %s", RAW_DATASET_DIR)
    logger.info("Output dataset: %s", PROCESSED_DATASET_DIR)
    logger.info("Target image size: %s", IMAGE_SIZE)

    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset does not exist: {RAW_DATASET_DIR}"
        )

    processed_count = 0
    failed_count = 0

    for split in sorted(EXPECTED_SPLITS):
        for class_name in sorted(EXPECTED_CLASSES):

            source_dir = RAW_DATASET_DIR / split / class_name
            destination_dir = (
                PROCESSED_DATASET_DIR / split / class_name
            )

            if not source_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: {source_dir}"
                )

            image_files = [
                path
                for path in source_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            ]

            logger.info(
                "Processing %s/%s: %d images",
                split,
                class_name,
                len(image_files),
            )

            for source_path in image_files:
                destination_path = (
                    destination_dir
                    / f"{source_path.stem}.jpg"
                )

                try:
                    preprocess_image(
                        source_path,
                        destination_path,
                    )

                    processed_count += 1

                except Exception as exc:
                    failed_count += 1

                    logger.error(
                        "Failed to process %s: %s",
                        source_path,
                        exc,
                    )

    return processed_count, failed_count


def main() -> None:
    try:
        processed_count, failed_count = preprocess_dataset()

        logger.info(
            "Images successfully processed: %d",
            processed_count,
        )

        logger.info(
            "Images failed: %d",
            failed_count,
        )

        if failed_count > 0:
            logger.error(
                "Preprocessing FAILED."
            )
            sys.exit(1)

        logger.info(
            "Preprocessing completed successfully."
        )

        sys.exit(0)

    except Exception:
        logger.exception(
            "Preprocessing service failed."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()