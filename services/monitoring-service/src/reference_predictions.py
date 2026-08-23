import time
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# PATHS
# ============================================================

SERVICE_ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = SERVICE_ROOT.parents[1]

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "dataset"
    / "train"
)

REFERENCE_DIR = (
    SERVICE_ROOT
    / "data"
    / "reference"
)

OUTPUT_FILE = (
    REFERENCE_DIR
    / "brainlens_reference_predictions.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_API_URL = (
    "http://localhost:8000/predict"
)

REFERENCE_MODE = True

TIMEOUT_SECONDS = 30

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
# BUILD DATASET
# ============================================================

def create_dataset_dataframe() -> pd.DataFrame:
    """
    Build the same dataset dataframe used by the training
    pipeline.
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
                        "actual_class": class_name,
                    }
                )

    dataframe = pd.DataFrame(records)

    if dataframe.empty:
        raise RuntimeError(
            "No training images found."
        )

    return dataframe


# ============================================================
# CREATE REFERENCE SPLIT
# ============================================================

def create_reference_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reproduce the exact 20% validation split used during
    BrainLens training.
    """

    from sklearn.model_selection import train_test_split

    _, validation_df = train_test_split(
        dataframe,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=dataframe["actual_class"],
    )

    return validation_df.reset_index(
        drop=True
    )


# ============================================================
# CALL PREDICTION API
# ============================================================

def predict_image(
    image_path: str,
) -> dict:
    """
    Send one image to the real prediction API in
    reference mode.
    """

    with open(
        image_path,
        "rb",
    ) as image_file:

        response = requests.post(
            PREDICTION_API_URL,
            params={
                "reference_mode": REFERENCE_MODE,
            },
            files={
                "file": (
                    Path(image_path).name,
                    image_file,
                    "image/jpeg",
                )
            },
            timeout=TIMEOUT_SECONDS,
        )

    response.raise_for_status()

    return response.json()


# ============================================================
# GENERATE REFERENCE PREDICTIONS
# ============================================================

def generate_reference_predictions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all reference images through the actual prediction API.
    """

    records = []

    total = len(dataframe)

    for index, row in dataframe.iterrows():

        image_path = row["path"]
        actual_class = row["actual_class"]

        print(
            f"[{index + 1}/{total}] "
            f"{Path(image_path).name}"
        )

        try:

            prediction = predict_image(
                image_path
            )

            records.append(
                {
                    "image_path": image_path,
                    "actual_class": actual_class,
                    "predicted_class": prediction[
                        "predicted_class"
                    ],
                    "confidence": prediction[
                        "confidence"
                    ],
                    "model_name": prediction[
                        "model_name"
                    ],
                    "model_alias": prediction[
                        "model_alias"
                    ],
                    "model_version": prediction[
                        "model_version"
                    ],
                    "model_uri": prediction[
                        "model_uri"
                    ],
                }
            )

        except Exception as exc:

            print(
                f"ERROR: {image_path}"
            )

            print(
                f"Reason: {exc}"
            )

            raise

    return pd.DataFrame(
        records
    )


# ============================================================
# SAVE
# ============================================================

def save_reference_predictions(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save reference predictions as Parquet.
    """

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nReference predictions saved to:"
        f"\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "Building BrainLens reference predictions..."
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
        f"Reference population size: "
        f"{len(reference_dataframe)}"
    )

    print(
        "\nReference class distribution:"
    )

    print(
        reference_dataframe[
            "actual_class"
        ]
        .value_counts()
        .sort_index()
    )

    reference_predictions = (
        generate_reference_predictions(
            reference_dataframe
        )
    )

    save_reference_predictions(
        reference_predictions
    )

    print(
        "\nPrediction distribution:"
    )

    print(
        reference_predictions[
            "predicted_class"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\nAverage confidence:"
    )

    print(
        reference_predictions[
            "confidence"
        ].mean()
    )


if __name__ == "__main__":
    main()