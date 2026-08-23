from pathlib import Path
import pandas as pd
from .features import extract_image_features


# ============================================================
# PATHS
# ============================================================

SERVICE_ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = SERVICE_ROOT.parents[1]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset"
)

TRAIN_DIR = DATASET_DIR / "train"

REFERENCE_DIR = (
    SERVICE_ROOT
    / "data"
    / "reference"
)

REFERENCE_FILE = (
    REFERENCE_DIR
    / "brainlens_reference_features.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

VALIDATION_SIZE = 0.20

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# BUILD DATAFRAME
# ============================================================

def create_dataset_dataframe() -> pd.DataFrame:
    """
    Build the dataset dataframe using the same class
    structure as the training pipeline.
    """

    records = []

    for class_name in CLASS_NAMES:

        class_dir = TRAIN_DIR / class_name

        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        for image_path in class_dir.rglob("*"):

            if (
                image_path.is_file()
                and image_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            ):
                records.append(
                    {
                        "path": str(image_path),
                        "class": class_name,
                    }
                )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "No training images were found."
        )

    return dataframe


# ============================================================
# CREATE REFERENCE SPLIT
# ============================================================

def create_reference_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the validation split used by the
    BrainLens training pipeline.
    """

    _, validation_df = (
        __import__(
            "sklearn.model_selection",
            fromlist=["train_test_split"],
        ).train_test_split(
            dataframe,
            test_size=VALIDATION_SIZE,
            random_state=SEED,
            stratify=dataframe["class"],
        )
    )

    return validation_df.reset_index(
        drop=True
    )


# ============================================================
# EXTRACT REFERENCE FEATURES
# ============================================================

def generate_reference_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract monitoring features from every image
    in the reference population.
    """

    records = []

    total = len(dataframe)

    for index, row in dataframe.iterrows():

        image_path = row["path"]
        actual_class = row["class"]

        features = extract_image_features(
            image_path
        )

        record = {
            "image_path": image_path,
            "actual_class": actual_class,
            **features,
        }

        records.append(record)

        if (index + 1) % 100 == 0:
            print(
                f"Processed {index + 1}/{total} images"
            )

    return pd.DataFrame(records)


# ============================================================
# SAVE REFERENCE DATASET
# ============================================================

def save_reference_dataset(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save the reference feature dataset as Parquet.
    """

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        REFERENCE_FILE,
        index=False,
    )

    print(
        f"Reference dataset saved to: "
        f"{REFERENCE_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Building BrainLens reference dataset..."
    )

    dataframe = create_dataset_dataframe()

    print(
        f"Total training images: {len(dataframe)}"
    )

    reference_dataframe = (
        create_reference_dataframe(
            dataframe
        )
    )

    print(
        "Reference population size: "
        f"{len(reference_dataframe)}"
    )

    print(
        "\nReference class distribution:"
    )

    print(
        reference_dataframe["class"]
        .value_counts()
        .sort_index()
    )

    reference_features = (
        generate_reference_features(
            reference_dataframe
        )
    )

    save_reference_dataset(
        reference_features
    )


if __name__ == "__main__":
    main()