"""
Gradio Web Interface for Waste Classifier
"""

import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from src.model_inference import WasteClassifier
from src.image_preprocessing import preprocess_image
from src.utils import get_waste_disposal_info

# Initialize classifier with default paths
MODEL_PATH = "models/waste_model.h5"
LABELS_PATH = "models/labels.txt"

try:
    classifier = WasteClassifier(MODEL_PATH, LABELS_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    classifier = None

def create_detailed_report(predicted_class: str, confidence: float, all_predictions: dict) -> str:
    """
    Create a detailed text report of classification results.
    
    Args:
        predicted_class: Name of the predicted class
        confidence: Confidence score for the predicted class
        all_predictions: Dictionary of all class probabilities
        
    Returns:
        str: Formatted report text
    """
    report = []
    report.append("=" * 60)
    report.append("WASTE CLASSIFICATION RESULTS")
    report.append("=" * 60)
    report.append("")
    report.append(f"🎯 PREDICTED CLASS: {predicted_class}")
    report.append(f"   Confidence: {confidence:.2%}")
    report.append("")
    report.append("📊 ALL CLASS PROBABILITIES:")
    report.append("-" * 60)
    
    # Sort by probability (descending)
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, probability in sorted_predictions:
        # Create a visual bar
        bar_length = int(probability * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        # Highlight the predicted class
        if class_name == predicted_class:
            report.append(f"  ➤ {class_name:20s} {bar} {probability:6.2%} ⭐")
        else:
            report.append(f"    {class_name:20s} {bar} {probability:6.2%}")
    
    report.append("=" * 60)
    report.append("")
    report.append("📋 DISPOSAL INFORMATION:")
    report.append("")
    report.append(get_waste_disposal_info(predicted_class))
    
    return "\n".join(report)

def predict_waste(image):
    """
    Predict the waste type from an image.
    
    Args:
        image: Input image (numpy array from Gradio)
        
    Returns:
        tuple: (dict of class probabilities, detailed report text)
    """
    if classifier is None:
        raise gr.Error("Model not loaded. Check console for errors.")
    
    if image is None:
        raise gr.Error("No image provided.")
        
    try:
        # Gradio provides RGB image, convert to BGR for preprocessing
        # Our preprocessing module expects BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Preprocess
        preprocessed = preprocess_image(image_bgr)
        
        # Predict
        results = classifier.predict_with_confidence(preprocessed)
        
        predicted_class = results['predicted_class']
        confidence = results['confidence']
        all_predictions = results['all_predictions']
        
        # Log results for debugging
        print(f"Predicted: {predicted_class} ({confidence:.2f})")
        print(f"All probs: {all_predictions}")
        
        # Create detailed report
        report = create_detailed_report(predicted_class, confidence, all_predictions)
        
        return all_predictions, report
        
    except Exception as e:
        print(f"Prediction error: {e}")
        raise gr.Error(f"Prediction error: {str(e)}")

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_waste,
    inputs=gr.Image(label="Upload Waste Image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Classification Result"),
        gr.Textbox(label="Detailed Report", lines=20, max_lines=30)
    ],
    title="♻️ Waste Classifier AI",
    description="Upload an image of waste to classify it into categories like Recyclable, Organic, Hazardous, etc.",
    examples=[]
)

if __name__ == "__main__":
    iface.launch()
