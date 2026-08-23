import logging

import numpy as np
from PIL import Image, ImageOps

import torch
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# RESIZE WITH PADDING
# ============================================================

class ResizeWithPadding:
    """
    Preserve image aspect ratio and pad to 224x224.

    This matches the preprocessing used during
    training and evaluation.
    """

    def __init__(
        self,
        size: int,
        fill: int = 0,
    ):
        self.size = size
        self.fill = fill

    def __call__(
        self,
        image: Image.Image,
    ) -> Image.Image:

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError(
                "Image dimensions must be greater than zero."
            )

        scale = min(
            self.size / width,
            self.size / height,
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
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

        return ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom,
            ),
            fill=self.fill,
        )


# ============================================================
# INFERENCE TRANSFORM
# ============================================================

inference_transform = transforms.Compose(
    [
        ResizeWithPadding(
            IMAGE_SIZE
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# MONITORING FEATURES
# ============================================================

def extract_monitoring_features(
    image: Image.Image,
) -> dict:
    """
    Extract monitoring features from the same deterministic
    image representation used before model inference.

    Processing:

        Original image
            ↓
        RGB conversion
            ↓
        ResizeWithPadding(224)
            ↓
        monitoring statistics

    ImageNet normalization is intentionally NOT applied to
    the monitoring features because we want interpretable
    values for brightness, contrast, and RGB statistics.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert(
        "RGB"
    )

    # --------------------------------------------------------
    # Original image information
    # --------------------------------------------------------

    original_width, original_height = (
        image.size
    )

    if original_width <= 0 or original_height <= 0:
        raise ValueError(
            "Image dimensions must be greater than zero."
        )

    original_aspect_ratio = (
        original_width / original_height
    )

    # --------------------------------------------------------
    # Apply the exact same deterministic resizing
    # used by inference.
    # --------------------------------------------------------

    processed_image = ResizeWithPadding(
        IMAGE_SIZE
    )(image)

    processed_width, processed_height = (
        processed_image.size
    )

    # --------------------------------------------------------
    # Convert processed image to NumPy.
    #
    # Values are initially 0-255.
    # --------------------------------------------------------

    image_array = np.asarray(
        processed_image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Normalize pixel values ONLY for statistics.
    #
    # This is NOT ImageNet normalization.
    #
    # It simply changes:
    #
    #     0-255 → 0-1
    #
    # so our monitoring features are easy to interpret.
    # --------------------------------------------------------

    normalized = (
        image_array / 255.0
    )

    # --------------------------------------------------------
    # RGB channels
    # --------------------------------------------------------

    red = normalized[:, :, 0]
    green = normalized[:, :, 1]
    blue = normalized[:, :, 2]

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    mean_r = float(
        red.mean()
    )

    mean_g = float(
        green.mean()
    )

    mean_b = float(
        blue.mean()
    )

    std_r = float(
        red.std()
    )

    std_g = float(
        green.std()
    )

    std_b = float(
        blue.std()
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        normalized.mean()
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    grayscale = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    contrast = float(
        grayscale.std()
    )

    return {
        "original_image_width": original_width,
        "original_image_height": original_height,
        "original_aspect_ratio": (
            original_aspect_ratio
        ),
        "processed_image_width": processed_width,
        "processed_image_height": processed_height,
        "mean_r": mean_r,
        "mean_g": mean_g,
        "mean_b": mean_b,
        "std_r": std_r,
        "std_g": std_g,
        "std_b": std_b,
        "brightness": brightness,
        "contrast": contrast,
    }


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert an input PIL image into a model-ready tensor.
    """

    if image is None:
        raise ValueError(
            "Image cannot be None."
        )

    image = image.convert(
        "RGB"
    )

    tensor = inference_transform(
        image
    )

    # Add batch dimension:
    #
    # [3, 224, 224]
    #       ↓
    # [1, 3, 224, 224]

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        device
    )