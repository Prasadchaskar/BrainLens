import logging
import sys
from pathlib import Path
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

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


DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "dataset"

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

def validate_structure()->bool:
    """Validate the expected dataset directort strcuture"""
    logger.info("Validating dataset structure...")
    logger.info("Dataset path: %s", DATASET_DIR)

    if not DATASET_DIR.exists():
        logger.error("Dataset directory does not exist.")
        return False
    
    actual_splits = {
        path.name for path in DATASET_DIR.iterdir()
        if path.is_dir()
    }

    if actual_splits != EXPECTED_SPLITS:
        logger.error(
            "Invalid dataset splits. Expected: %s, Found: %s",
            sorted(EXPECTED_SPLITS),
            sorted(actual_splits),
        )
        return False
    valid = True
    for split in sorted(EXPECTED_SPLITS):
        split_path = DATASET_DIR / split
        actual_classes = {
            path.name
            for path in split_path.iterdir()
            if path.is_dir()
        }

        if actual_classes != EXPECTED_CLASSES:
            logger.error(
                "Invalid classes in %s. Expected: %s, Found: %s",
                split,
                sorted(EXPECTED_CLASSES),
                sorted(actual_classes),
            )
            valid = False
    return valid

def validate_images() -> tuple[bool, int, int]:
    """Validate image files and detect corrupted images."""

    logger.info("Validating image files...")

    total_images = 0
    corrupted_images = 0

    for split in sorted(EXPECTED_SPLITS):
        for class_name in sorted(EXPECTED_CLASSES):

            class_dir = DATASET_DIR / split / class_name

            image_files = [
                path
                for path in class_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            logger.info(
                "%s/%s: %d images",
                split,
                class_name,
                len(image_files),
            )

            if not image_files:
                logger.error(
                    "No images found in %s/%s",
                    split,
                    class_name,
                )

            for image_path in image_files:
                total_images += 1

                try:
                    with Image.open(image_path) as image:
                        image.verify()

                except Exception as exc:
                    corrupted_images += 1

                    logger.error(
                        "Corrupted/unreadable image: %s | %s",
                        image_path,
                        exc,
                    )

    return (
        corrupted_images == 0,
        total_images,
        corrupted_images,
    )

def validate_dataset() -> bool:
    """Run all dataset validation checks."""

    logger.info("Starting dataset validation...")

    structure_valid = validate_structure()

    if not structure_valid:
        return False

    images_valid, total_images, corrupted_images = validate_images()

    logger.info("Total images checked: %d", total_images)
    logger.info("Corrupted images: %d", corrupted_images)

    return images_valid


def main() -> None:
    try:
        is_valid = validate_dataset()

        if is_valid:
            logger.info("Dataset validation PASSED.")
            sys.exit(0)

        logger.error("Dataset validation FAILED.")
        sys.exit(1)

    except Exception:
        logger.exception("Data validation service failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()