"""
Image Preprocessing Module
Handles image preprocessing for the MobileNetV2 waste classification model.
"""

import cv2
import numpy as np
from typing import Literal, Tuple


Normalization = Literal["zero_one", "mobilenet_v2"]


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalization: Normalization = "zero_one",
) -> np.ndarray:
    """
    Preprocess an image for the MobileNetV2 model.
    
    The preprocessing steps match the training pipeline:
    1. Resize to target size (224x224)
    2. Convert BGR to RGB
    3. Normalize pixel values to [0, 1] or MobileNetV2's [-1, 1] range
    4. Add batch dimension
    
    Args:
        image: Input image in BGR format (OpenCV format)
        target_size: Target size for the model (default: (224, 224))
        normalization: ``zero_one`` for the legacy shipped model, or
            ``mobilenet_v2`` for models produced by ``train.py``.
    
    Returns:
        numpy.ndarray: Preprocessed image ready for model inference
                      Shape: (1, 224, 224, 3)
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a non-empty BGR NumPy array with shape (H, W, 3)")
    if image.size == 0:
        raise ValueError("image must not be empty")
    if normalization not in ("zero_one", "mobilenet_v2"):
        raise ValueError(f"Unsupported normalization mode: {normalization}")

    # Resize image to target size
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Keep this coupled to the training pipeline. The existing .h5 model was
    # trained with /255; newly trained MobileNetV2 models use [-1, 1].
    normalized = rgb_image.astype(np.float32)
    if normalization == "zero_one":
        normalized /= 255.0
    else:
        normalized = (normalized / 127.5) - 1.0
    
    # Add batch dimension (model expects batch of images)
    batched = np.expand_dims(normalized, axis=0)
    
    return batched


def visualize_preprocessed_image(
    preprocessed_image: np.ndarray, normalization: Normalization = "zero_one"
) -> np.ndarray:
    """
    Convert preprocessed image back to displayable format.
    
    Args:
        preprocessed_image: Preprocessed image with shape (1, 224, 224, 3)
    
    Returns:
        numpy.ndarray: Image in BGR format suitable for display with OpenCV
    """
    # Remove batch dimension
    image = preprocessed_image[0]
    
    if normalization == "zero_one":
        image = image * 255
    elif normalization == "mobilenet_v2":
        image = (image + 1.0) * 127.5
    else:
        raise ValueError(f"Unsupported normalization mode: {normalization}")
    image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV display
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return bgr_image
