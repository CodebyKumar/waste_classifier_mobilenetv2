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

def get_class_info(waste_class: str) -> str:
    """
    Get detailed information about each waste class.
    """
    class_info = {
        "Hazardous": {
            "icon": "⚠️",
            "description": "Materials that pose risks to health or environment",
            "examples": "Batteries, chemicals, paint, electronics, fluorescent bulbs, pesticides",
            "disposal": "Take to designated hazardous waste collection center. Never mix with regular trash.",
            "tips": "Store in original containers, keep away from children and pets"
        },
        "Non-Recyclable": {
            "icon": "🗑️",
            "description": "Materials that cannot be recycled through standard programs",
            "examples": "Styrofoam, plastic bags, chip bags, contaminated materials, ceramics",
            "disposal": "Dispose in general waste bin. Consider reducing usage of these items.",
            "tips": "Look for alternative products with recyclable packaging"
        },
        "Organic": {
            "icon": "🌱",
            "description": "Biodegradable materials from living organisms",
            "examples": "Food scraps, yard waste, paper towels, coffee grounds, eggshells",
            "disposal": "Compost at home or use green waste bin. Keep separate from other waste.",
            "tips": "Composting reduces landfill waste and creates nutrient-rich soil"
        },
        "Recyclable": {
            "icon": "♻️",
            "description": "Materials that can be processed and reused",
            "examples": "Paper, cardboard, glass bottles, aluminum cans, plastic bottles (check number)",
            "disposal": "Place in recycling bin. Clean and dry items before recycling.",
            "tips": "Check local recycling guidelines for accepted materials"
        }
    }
    
    info = class_info.get(waste_class, {})
    if not info:
        return "No information available."
    
    return f"""{info['icon']} {waste_class.upper()} WASTE

Description: {info['description']}

Examples: {info['examples']}

Disposal Instructions:
{info['disposal']}

Helpful Tips:
{info['tips']}"""

def create_detailed_report(predicted_class: str, confidence: float, all_predictions: dict) -> str:
    """
    Create a detailed text report of classification results.
    """
    report = []
    report.append("=" * 60)
    report.append("WASTE CLASSIFICATION RESULTS")
    report.append("=" * 60)
    report.append("")
    report.append(f"PREDICTED CLASS: {predicted_class}")
    report.append(f"Confidence: {confidence:.2%}")
    report.append("")
    report.append("ALL CLASS PROBABILITIES:")
    report.append("-" * 60)
    
    # Sort by probability (descending)
    sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, probability in sorted_predictions:
        # Create a visual bar
        bar_length = int(probability * 40)
        bar = "█" * bar_length + "░" * (40 - bar_length)
        
        # Highlight the predicted class
        if class_name == predicted_class:
            report.append(f"  > {class_name:20s} {bar} {probability:6.2%} (PREDICTED)")
        else:
            report.append(f"    {class_name:20s} {bar} {probability:6.2%}")
    
    report.append("=" * 60)
    report.append("")
    report.append(get_class_info(predicted_class))
    
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
    title="🌍 Smart Waste Classifier",
    description="Upload an image of waste to classify it into categories: Recyclable, Organic, Hazardous, or Non-Recyclable. Get instant disposal instructions and helpful tips!",
    examples=[]
)

if __name__ == "__main__":
    iface.launch()
