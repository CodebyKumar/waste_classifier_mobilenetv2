"""
Image Capture Module
Provides functions to capture images from camera or load from file path.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional


def capture_from_camera(camera_index: int = 0, preview: bool = True) -> Optional[np.ndarray]:
    """
    Capture a photo using the camera.
    
    Args:
        camera_index: Index of the camera to use (default: 0 for primary camera)
        preview: Whether to show a preview window (default: True)
    
    Returns:
        numpy.ndarray: Captured image in BGR format, or None if capture failed
    
    Usage:
        Press 'SPACE' to capture the image
        Press 'q' or 'ESC' to quit without capturing
    """
    # Initialize camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        return None
    
    print("Camera opened successfully!")
    print("Press SPACE to capture image")
    print("Press 'q' or ESC to quit")
    
    captured_image = None
    
    while True:
        # Read frame from camera
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        if preview:
            # Display the frame
            cv2.imshow('Camera Preview - Press SPACE to capture', frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # SPACE key to capture
        if key == ord(' '):
            captured_image = frame.copy()
            print("Image captured!")
            break
        
        # 'q' or ESC to quit
        elif key == ord('q') or key == 27:
            print("Capture cancelled")
            break
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    return captured_image


def load_from_path(image_path: str) -> Optional[np.ndarray]:
    """
    Load an image from the specified file path.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        numpy.ndarray: Loaded image in BGR format, or None if loading failed
    """
    # Convert to Path object for better path handling
    path = Path(image_path)
    
    # Check if file exists
    if not path.exists():
        print(f"Error: Image file not found at {image_path}")
        return None
    
    # Check if it's a file (not a directory)
    if not path.is_file():
        print(f"Error: {image_path} is not a file")
        return None
    
    # Load the image
    image = cv2.imread(str(path))
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Supported formats: .jpg, .jpeg, .png, .bmp, .tiff")
        return None
    
    print(f"Image loaded successfully from {image_path}")
    print(f"Image shape: {image.shape}")
    
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Save an image to the specified path.
    
    Args:
        image: Image array in BGR format
        output_path: Path where the image should be saved
    
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Image saved to {output_path}")
        else:
            print(f"Failed to save image to {output_path}")
        return success
    except Exception as e:
        print(f"Error saving image: {e}")
        return False
