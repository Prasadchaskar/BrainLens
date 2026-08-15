import logging
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")

if PROJECT_ROOT_ENV:
    PROJECT_ROOT = Path(PROJECT_ROOT_ENV)
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ============================================================
# PATHS
# ============================================================

RAW_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset"
)

PROCESSED_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dataset"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

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


# ============================================================
# RESIZE WITH PADDING
# ============================================================

class ResizeWithPadding:
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a square canvas.

    This matches the preprocessing used in the
    Brain_Tumor_Exp.ipynb research pipeline.
    """

    def __init__(
        self,
        size: int,
        fill=0,
    ):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:

        width, height = image.size

        scale = min(
            self.size / width,
            self.size / height,
        )

        new_width = int(width * scale)
        new_height = int(height * scale)

        image = image.resize(
            (new_width, new_height),
            Image.Resampling.BILINEAR,
        )

        pad_left = (
            self.size - new_width
        ) // 2

        pad_top = (
            self.size - new_height
        ) // 2

        pad_right = (
            self.size
            - new_width
            - pad_left
        )

        pad_bottom = (
            self.size
            - new_height
            - pad_top
        )

        image = ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ),
            fill=self.fill,
        )

        return image


RESIZE_WITH_PADDING = ResizeWithPadding(
    IMAGE_SIZE
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(
    source_path: Path,
    destination_path: Path,
) -> None:
    """
    Convert image to RGB and apply aspect-ratio-preserving
    resize with padding to 224x224.
    """

    with Image.open(source_path) as image:

        image = image.convert("RGB")

        image = RESIZE_WITH_PADDING(image)

        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            destination_path,
            format="JPEG",
        )


# ============================================================
# DATASET PREPROCESSING
# ============================================================

def preprocess_dataset() -> tuple[int, int]:
    """
    Preprocess the complete raw dataset.

    The directory structure and class membership are preserved.
    """

    logger.info(
        "Starting dataset preprocessing..."
    )

    logger.info(
        "Input dataset: %s",
        RAW_DATASET_DIR,
    )

    logger.info(
        "Output dataset: %s",
        PROCESSED_DATASET_DIR,
    )

    logger.info(
        "Target image size: (%d, %d)",
        IMAGE_SIZE,
        IMAGE_SIZE,
    )

    if not RAW_DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset does not exist: "
            f"{RAW_DATASET_DIR}"
        )

    processed_count = 0
    failed_count = 0

    for split in sorted(EXPECTED_SPLITS):

        for class_name in sorted(
            EXPECTED_CLASSES
        ):

            source_dir = (
                RAW_DATASET_DIR
                / split
                / class_name
            )

            destination_dir = (
                PROCESSED_DATASET_DIR
                / split
                / class_name
            )

            if not source_dir.exists():
                raise FileNotFoundError(
                    f"Missing directory: "
                    f"{source_dir}"
                )

            image_files = [
                path
                for path in source_dir.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
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

    return (
        processed_count,
        failed_count,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    try:

        (
            processed_count,
            failed_count,
        ) = preprocess_dataset()

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