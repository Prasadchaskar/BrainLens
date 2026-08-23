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
    / "brainlens_reference_features.parquet"
)

TEST_DATA_DIR = (
    SERVICE_ROOT
    / "data"
    / "test"
)

DRIFTED_FILE = (
    TEST_DATA_DIR
    / "drifted_current.parquet"
)


# ============================================================
# CREATE DRIFTED DATA
# ============================================================

def create_drifted_dataset() -> None:
    """
    Create a synthetic production-like dataset with
    deliberately shifted image statistics.

    The original reference dataset is never modified.
    """

    reference = pd.read_parquet(
        REFERENCE_FILE
    )

    current = reference.copy()

    # --------------------------------------------------------
    # Introduce deliberate distribution shifts
    # --------------------------------------------------------

    current["brightness"] = (
        current["brightness"] * 1.5
    ).clip(upper=1.0)

    current["contrast"] = (
        current["contrast"] * 1.5
    )

    current["mean_r"] = (
        current["mean_r"] * 1.4
    ).clip(upper=1.0)

    # --------------------------------------------------------
    # Save synthetic current dataset
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
        f"Drifted dataset created: {DRIFTED_FILE}"
    )

    print(
        f"Rows: {len(current)}"
    )


if __name__ == "__main__":
    create_drifted_dataset()