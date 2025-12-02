"""
Model Inference Module
Handles loading the trained model and making predictions.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow import keras


class WasteClassifier:
    """
    Waste classification model wrapper.
    Loads the trained MobileNetV2 model and provides prediction functionality.
    """
    
    def __init__(self, model_path: str, labels_path: str):
        """
        Initialize the waste classifier.
        
        Args:
            model_path: Path to the trained model (.h5 file)
            labels_path: Path to the labels file (.txt file)
        """
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)
        self.model = None
        self.class_labels = []
        
        # Load model and labels
        self._load_model()
        self._load_labels()
    
    def _load_model(self):
        """Load the trained Keras model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(str(self.model_path))
        print("Model loaded successfully!")
        print(f"Model input shape: {self.model.input_shape}")
        print(f"Model output shape: {self.model.output_shape}")
    
    def _load_labels(self):
        """Load class labels from file."""
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        
        with open(self.labels_path, 'r') as f:
            self.class_labels = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"Loaded {len(self.class_labels)} class labels: {self.class_labels}")
    
    def predict(self, preprocessed_image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """
        Make a prediction on a preprocessed image.
        
        Args:
            preprocessed_image: Preprocessed image with shape (1, 224, 224, 3)
        
        Returns:
            Tuple containing:
                - predicted_class: Name of the predicted class
                - all_predictions: Dictionary mapping class names to probabilities
        """
        # Make prediction
        predictions = self.model.predict(preprocessed_image, verbose=0)
        
        # Get probabilities for all classes
        probabilities = predictions[0]
        
        # Create dictionary of class names to probabilities
        all_predictions = {
            self.class_labels[i]: float(probabilities[i])
            for i in range(len(self.class_labels))
        }
        
        # Get the predicted class (highest probability)
        predicted_index = np.argmax(probabilities)
        predicted_class = self.class_labels[predicted_index]
        
        return predicted_class, all_predictions
    
    def predict_with_confidence(self, preprocessed_image: np.ndarray) -> Dict:
        """
        Make a prediction and return detailed results.
        
        Args:
            preprocessed_image: Preprocessed image with shape (1, 224, 224, 3)
        
        Returns:
            Dictionary containing:
                - predicted_class: Name of the predicted class
                - confidence: Confidence score for the predicted class
                - all_predictions: Dictionary of all class probabilities
                - sorted_predictions: List of (class, probability) tuples sorted by probability
        """
        predicted_class, all_predictions = self.predict(preprocessed_image)
        
        # Sort predictions by probability (descending)
        sorted_predictions = sorted(
            all_predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'predicted_class': predicted_class,
            'confidence': all_predictions[predicted_class],
            'all_predictions': all_predictions,
            'sorted_predictions': sorted_predictions
        }
