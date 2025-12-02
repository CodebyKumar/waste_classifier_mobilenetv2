"""
Utility Functions
Helper functions for displaying results and formatting output.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple


def display_results(image: np.ndarray, predicted_class: str, all_predictions: Dict[str, float], 
                   window_name: str = "Waste Classification Result"):
    """
    Display the image with prediction results overlaid.
    
    Args:
        image: Original image in BGR format
        predicted_class: Name of the predicted class
        all_predictions: Dictionary of all class probabilities
        window_name: Name of the display window
    """
    # Create a copy to avoid modifying the original
    display_img = image.copy()
    
    # Resize for better display if image is too large
    height, width = display_img.shape[:2]
    if height > 800 or width > 800:
        scale = min(800 / height, 800 / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        display_img = cv2.resize(display_img, (new_width, new_height))
    
    # Add text overlay with prediction
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # Main prediction
    text = f"Predicted: {predicted_class}"
    confidence = all_predictions[predicted_class]
    text_conf = f"Confidence: {confidence:.2%}"
    
    # Calculate text size for background rectangle
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw semi-transparent background
    overlay = display_img.copy()
    cv2.rectangle(overlay, (10, 10), (text_width + 30, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0, display_img)
    
    # Draw text
    cv2.putText(display_img, text, (20, 40), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(display_img, text_conf, (20, 75), font, font_scale, (0, 255, 0), thickness)
    
    # Display the image
    cv2.imshow(window_name, display_img)
    print("\nPress any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def print_prediction_results(predicted_class: str, all_predictions: Dict[str, float]):
    """
    Print formatted prediction results to console.
    
    Args:
        predicted_class: Name of the predicted class
        all_predictions: Dictionary of all class probabilities
    """
    print("\n" + "="*60)
    print("WASTE CLASSIFICATION RESULTS")
    print("="*60)
    
    print(f"\n🎯 PREDICTED CLASS: {predicted_class}")
    print(f"   Confidence: {all_predictions[predicted_class]:.2%}")
    
    print("\n📊 ALL CLASS PROBABILITIES:")
    print("-" * 60)
    
    # Sort by probability (descending)
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, probability in sorted_predictions:
        # Create a visual bar
        bar_length = int(probability * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        # Highlight the predicted class
        if class_name == predicted_class:
            print(f"  ➤ {class_name:20s} {bar} {probability:6.2%} ⭐")
        else:
            print(f"    {class_name:20s} {bar} {probability:6.2%}")
    
    print("="*60 + "\n")


def create_prediction_summary(predicted_class: str, confidence: float, 
                              all_predictions: Dict[str, float]) -> str:
    """
    Create a text summary of the prediction results.
    
    Args:
        predicted_class: Name of the predicted class
        confidence: Confidence score for the predicted class
        all_predictions: Dictionary of all class probabilities
    
    Returns:
        str: Formatted summary text
    """
    summary = []
    summary.append("WASTE CLASSIFICATION SUMMARY")
    summary.append("=" * 50)
    summary.append(f"Predicted Class: {predicted_class}")
    summary.append(f"Confidence: {confidence:.2%}")
    summary.append("\nAll Predictions:")
    
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    for class_name, prob in sorted_predictions:
        summary.append(f"  {class_name}: {prob:.2%}")
    
    return "\n".join(summary)


def get_waste_disposal_info(waste_class: str) -> str:
    """
    Get disposal information for the predicted waste class.
    
    Args:
        waste_class: Name of the waste class
    
    Returns:
        str: Disposal information and recommendations
    """
    disposal_info = {
        "Hazardous": """
        ⚠️  HAZARDOUS WASTE
        - Requires special handling and disposal
        - Do not mix with regular trash
        - Take to designated hazardous waste collection center
        - Examples: batteries, chemicals, paint, electronics
        """,
        "Non-Recyclable": """
        🗑️  NON-RECYCLABLE WASTE
        - Dispose in general waste bin
        - Cannot be recycled through standard programs
        - Consider reducing usage of such items
        - Examples: certain plastics, contaminated materials
        """,
        "Organic": """
        🌱  ORGANIC WASTE
        - Can be composted
        - Biodegradable material
        - Keep separate from other waste types
        - Examples: food scraps, yard waste, paper
        """,
        "Recyclable": """
        ♻️  RECYCLABLE WASTE
        - Place in recycling bin
        - Clean and dry before recycling
        - Check local recycling guidelines
        - Examples: paper, cardboard, certain plastics, glass, metal
        """
    }
    
    return disposal_info.get(waste_class, "No disposal information available.")
