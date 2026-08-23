from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

SERVICE_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_FILE = (
    SERVICE_ROOT
    / "data"
    / "reference"
    / "brainlens_reference_predictions.parquet"
)

TEST_DATA_DIR = (
    SERVICE_ROOT
    / "data"
    / "test"
)

DRIFTED_FILE = (
    TEST_DATA_DIR
    / "drifted_prediction_current.parquet"
)


# ============================================================
# CREATE DRIFTED PREDICTION DATA
# ============================================================

def create_drifted_prediction_dataset() -> None:
    """
    Create a synthetic current prediction dataset with
    deliberately changed model-output distributions.

    The original reference dataset is never modified.
    """

    reference = pd.read_parquet(
        REFERENCE_FILE
    )

    current = reference.copy()

    # --------------------------------------------------------
    # Deliberately change the prediction distribution.
    #
    # We create a strongly meningioma-heavy population.
    # --------------------------------------------------------

    current["predicted_class"] = (
        [
            "meningioma"
        ]
        * len(current)
    )

    # --------------------------------------------------------
    # Deliberately reduce confidence.
    #
    # Reference average confidence is approximately 0.937.
    # We create a lower-confidence production-like population.
    # --------------------------------------------------------

    current["confidence"] = (
        current["confidence"] * 0.75
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # --------------------------------------------------------
    # Save synthetic current dataset.
    # --------------------------------------------------------

    TEST_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    current.to_parquet(
        DRIFTED_FILE,
        index=False,
    )

    print(
        f"Drifted prediction dataset created: "
        f"{DRIFTED_FILE}"
    )

    print(
        f"Rows: {len(current)}"
    )

    print(
        "\nReference prediction distribution:"
    )

    print(
        reference[
            "predicted_class"
        ]
        .value_counts(
            normalize=True
        )
    )

    print(
        "\nDrifted prediction distribution:"
    )

    print(
        current[
            "predicted_class"
        ]
        .value_counts(
            normalize=True
        )
    )

    print(
        "\nReference average confidence:"
    )

    print(
        reference[
            "confidence"
        ].mean()
    )

    print(
        "\nDrifted average confidence:"
    )

    print(
        current[
            "confidence"
        ].mean()
    )


if __name__ == "__main__":
    create_drifted_prediction_dataset()