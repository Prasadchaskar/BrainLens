import pandas as pd


LOW_CONFIDENCE_THRESHOLD = 0.70


def calculate_prediction_metrics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate basic metrics about model predictions.

    These metrics do not require ground-truth labels.
    """

    if dataframe.empty:
        return {
            "prediction_count": 0,
            "average_confidence": 0.0,
            "low_confidence_rate": 0.0,
        }

    confidence = pd.to_numeric(
        dataframe["confidence"],
        errors="coerce",
    )

    prediction_count = len(dataframe)

    average_confidence = float(
        confidence.mean()
    )

    low_confidence_rate = float(
        (
            confidence < LOW_CONFIDENCE_THRESHOLD
        ).mean()
    )

    return {
        "prediction_count": prediction_count,
        "average_confidence": average_confidence,
        "low_confidence_rate": low_confidence_rate,
    }


def calculate_prediction_distribution(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate the percentage of predictions belonging
    to each predicted class.
    """

    if dataframe.empty:
        return {}

    distribution = (
        dataframe["predicted_class"]
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    return {
        str(label): float(value)
        for label, value in distribution.items()
    }


def calculate_data_quality_metrics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate average and standard deviation for the
    image features captured during prediction.
    """

    if dataframe.empty:
        return {}

    feature_columns = [
        "image_width",
        "image_height",
        "aspect_ratio",
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "brightness",
        "contrast",
    ]

    metrics = {}

    for column in feature_columns:

        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

        metrics[f"{column}_mean"] = float(
            values.mean()
        )

        metrics[f"{column}_std"] = float(
            values.std()
        )

    return metrics


def calculate_model_performance(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Calculate model performance using predictions for
    which ground truth is available.
    """

    labeled = dataframe[
        dataframe["actual_class"].notna()
        & (
            dataframe["actual_class"]
            != ""
        )
    ].copy()

    if labeled.empty:
        return {
            "labeled_predictions": 0,
        }

    correct = (
        labeled["predicted_class"]
        == labeled["actual_class"]
    ).sum()

    total = len(labeled)

    accuracy = correct / total

    return {
        "labeled_predictions": total,
        "correct_predictions": int(correct),
        "accuracy": float(accuracy),
    }