import logging

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

    This must match the preprocessing used during
    training and evaluation.
    """

    def __init__(
        self,
        size: int,
        fill=0,
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
    # [3, 224, 224] -> [1, 3, 224, 224]
    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(
        device
    )