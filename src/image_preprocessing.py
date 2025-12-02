"""
Image Preprocessing Module
Handles image preprocessing for the MobileNetV2 waste classification model.
"""

import cv2
import numpy as np
from typing import Tuple


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Preprocess an image for the MobileNetV2 model.
    
    The preprocessing steps match the training pipeline:
    1. Resize to target size (224x224)
    2. Convert BGR to RGB
    3. Normalize pixel values to [0, 1]
    4. Add batch dimension
    
    Args:
        image: Input image in BGR format (OpenCV format)
        target_size: Target size for the model (default: (224, 224))
    
    Returns:
        numpy.ndarray: Preprocessed image ready for model inference
                      Shape: (1, 224, 224, 3)
    """
    # Resize image to target size
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB (OpenCV loads as BGR, but model expects RGB)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    normalized = rgb_image.astype(np.float32) / 255.0
    
    # Add batch dimension (model expects batch of images)
    batched = np.expand_dims(normalized, axis=0)
    
    return batched


def visualize_preprocessed_image(preprocessed_image: np.ndarray) -> np.ndarray:
    """
    Convert preprocessed image back to displayable format.
    
    Args:
        preprocessed_image: Preprocessed image with shape (1, 224, 224, 3)
    
    Returns:
        numpy.ndarray: Image in BGR format suitable for display with OpenCV
    """
    # Remove batch dimension
    image = preprocessed_image[0]
    
    # Denormalize from [0, 1] to [0, 255]
    image = (image * 255).astype(np.uint8)
    
    # Convert RGB back to BGR for OpenCV display
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return bgr_image
