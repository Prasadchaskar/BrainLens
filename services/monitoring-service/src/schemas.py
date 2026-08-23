from datetime import datetime, timezone
from typing import Optional
from typing import Literal
from pydantic import BaseModel, Field


class PredictionEvent(BaseModel):
    """
    Monitoring event generated for every model prediction.
    """

    # --------------------------------------------------------
    # Prediction identification
    # --------------------------------------------------------

    prediction_id: str = Field(
        ...,
        description="Unique identifier for the prediction event",
    )

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the prediction was made",
    )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    model_name: str

    model_alias: str

    model_version: str

    model_uri: str

    # --------------------------------------------------------
    # Prediction information
    # --------------------------------------------------------

    predicted_class: str

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence between 0 and 1",
    )

    # --------------------------------------------------------
    # Original image information
    # --------------------------------------------------------

    original_image_width: int = Field(
        ...,
        gt=0,
        description="Width of the original input image",
    )

    original_image_height: int = Field(
        ...,
        gt=0,
        description="Height of the original input image",
    )

    original_aspect_ratio: float = Field(
        ...,
        gt=0,
        description="Width divided by height of the original image",
    )

    # --------------------------------------------------------
    # Processed image information
    # --------------------------------------------------------

    processed_image_width: int = Field(
        ...,
        gt=0,
        description="Width after BrainLens preprocessing",
    )

    processed_image_height: int = Field(
        ...,
        gt=0,
        description="Height after BrainLens preprocessing",
    )

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    mean_r: float

    mean_g: float

    mean_b: float

    std_r: float

    std_g: float

    std_b: float

    # --------------------------------------------------------
    # Image quality / statistical features
    # --------------------------------------------------------

    brightness: float

    contrast: float

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    actual_class: Optional[str] = None


class GroundTruthUpdate(BaseModel):
    """
    Ground-truth label received after the original prediction.
    """

    actual_class: Literal[
        "glioma",
        "meningioma",
        "no_tumor",
    ]

    def validate_class(self) -> None:
        """
        Validate that the ground-truth class is one of
        the supported BrainLens classes.
        """

        allowed_classes = {
            "glioma",
            "meningioma",
            "no_tumor",
        }

        if self.actual_class not in allowed_classes:
            raise ValueError(
                "actual_class must be one of: "
                "glioma, meningioma, no_tumor"
            )