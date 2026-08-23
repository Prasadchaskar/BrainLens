from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps


ImageInput = Union[
    Image.Image,
    str,
    Path,
]

IMAGE_SIZE = 224


def load_image(
    image: ImageInput,
) -> Image.Image:
    """
    Load an image from a path or return an already-loaded PIL image.
    """

    if isinstance(image, Image.Image):
        return image

    return Image.open(image)


class ResizeWithPadding:
    """
    Resize an image while preserving its aspect ratio,
    then pad it to a square.

    This matches the preprocessing used by the BrainLens
    training/validation pipeline.
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


def extract_image_features(
    image: ImageInput,
) -> dict:
    """
    Extract monitoring features using the same deterministic
    preprocessing used before model inference.

    Processing:

        Original image
            ↓
        RGB conversion
            ↓
        ResizeWithPadding(224)
            ↓
        feature extraction
    """

    image = load_image(image)

    # --------------------------------------------------------
    # Convert to RGB
    # --------------------------------------------------------

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Preserve original image information
    # --------------------------------------------------------

    original_width, original_height = image.size

    if original_height <= 0:
        raise ValueError(
            "Image height cannot be zero."
        )

    original_aspect_ratio = (
        original_width / original_height
    )

    # --------------------------------------------------------
    # Apply the same deterministic preprocessing used by
    # validation/inference.
    # --------------------------------------------------------

    processed_image = ResizeWithPadding(
        IMAGE_SIZE
    )(image)

    processed_width, processed_height = (
        processed_image.size
    )

    # --------------------------------------------------------
    # Convert processed image to NumPy
    #
    # Values remain in 0-255 at this point.
    # --------------------------------------------------------

    image_array = np.asarray(
        processed_image,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Normalize only for feature calculation.
    #
    # IMPORTANT:
    # We are NOT applying ImageNet mean/std normalization.
    #
    # We want interpretable image statistics such as:
    # brightness = 0.42
    # contrast   = 0.18
    # --------------------------------------------------------

    normalized = image_array / 255.0

    red = normalized[:, :, 0]
    green = normalized[:, :, 1]
    blue = normalized[:, :, 2]

    # --------------------------------------------------------
    # RGB statistics
    # --------------------------------------------------------

    mean_r = float(red.mean())
    mean_g = float(green.mean())
    mean_b = float(blue.mean())

    std_r = float(red.std())
    std_g = float(green.std())
    std_b = float(blue.std())

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        normalized.mean()
    )

    # --------------------------------------------------------
    # Contrast
    #
    # Convert RGB to grayscale and use its standard deviation.
    # --------------------------------------------------------

    grayscale = (
        0.299 * red
        + 0.587 * green
        + 0.114 * blue
    )

    contrast = float(
        grayscale.std()
    )

    # --------------------------------------------------------
    # Return monitoring features
    # --------------------------------------------------------

    return {
        "original_image_width": original_width,
        "original_image_height": original_height,
        "original_aspect_ratio": original_aspect_ratio,
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