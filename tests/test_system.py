"""
Test Script for Waste Classification System
Verifies that all modules are working correctly.
"""

import sys
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import cv2
        import tensorflow as tf
        import numpy as np
        print("✓ All required packages imported successfully")
        print(f"  - OpenCV version: {cv2.__version__}")
        print(f"  - TensorFlow version: {tf.__version__}")
        print(f"  - NumPy version: {np.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_custom_modules():
    """Test that all custom modules can be imported."""
    print("\nTesting custom modules...")
    try:
        from src import image_capture
        from src import image_preprocessing
        from src import model_inference
        from src import utils
        print("✓ All custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_preprocessing():
    """Test image preprocessing functions."""
    print("\nTesting image preprocessing...")
    try:
        from src.image_preprocessing import preprocess_image, visualize_preprocessed_image
        
        # Create a dummy image (224x224x3)
        dummy_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        
        # Test preprocessing
        preprocessed = preprocess_image(dummy_image)
        
        # Verify shape
        assert preprocessed.shape == (1, 224, 224, 3), f"Expected shape (1, 224, 224, 3), got {preprocessed.shape}"
        
        # Verify normalization (values should be in [0, 1])
        assert preprocessed.min() >= 0 and preprocessed.max() <= 1, "Values not normalized to [0, 1]"
        
        # Test visualization
        visualized = visualize_preprocessed_image(preprocessed)
        assert visualized.shape == (224, 224, 3), f"Expected shape (224, 224, 3), got {visualized.shape}"
        
        print("✓ Image preprocessing works correctly")
        print(f"  - Input shape: {dummy_image.shape}")
        print(f"  - Preprocessed shape: {preprocessed.shape}")
        print(f"  - Value range: [{preprocessed.min():.3f}, {preprocessed.max():.3f}]")
        return True
    except Exception as e:
        print(f"✗ Preprocessing test failed: {e}")
        return False


def test_model_loading():
    """Test model loading (if model file exists)."""
    print("\nTesting model loading...")
    try:
        from src.model_inference import WasteClassifier
        
        model_path = "models/waste_model.h5"
        labels_path = "models/labels.txt"
        
        if not Path(model_path).exists():
            print(f"⚠ Model file not found at {model_path} - skipping model test")
            return True
        
        if not Path(labels_path).exists():
            print(f"⚠ Labels file not found at {labels_path} - skipping model test")
            return True
        
        # Load classifier
        classifier = WasteClassifier(model_path, labels_path)
        
        # Create dummy preprocessed image
        dummy_preprocessed = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # Test prediction
        results = classifier.predict_with_confidence(dummy_preprocessed)
        
        # Verify results structure
        assert 'predicted_class' in results
        assert 'confidence' in results
        assert 'all_predictions' in results
        assert 'sorted_predictions' in results
        
        print("✓ Model loading and prediction works correctly")
        print(f"  - Model loaded from: {model_path}")
        print(f"  - Number of classes: {len(classifier.class_labels)}")
        print(f"  - Classes: {classifier.class_labels}")
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    try:
        from src.utils import print_prediction_results, get_waste_disposal_info, create_prediction_summary
        
        # Test data
        predicted_class = "Recyclable"
        all_predictions = {
            "Hazardous": 0.05,
            "Non-Recyclable": 0.10,
            "Organic": 0.15,
            "Recyclable": 0.70
        }
        
        # Test print function (just check it doesn't crash)
        print("\n--- Testing print_prediction_results ---")
        print_prediction_results(predicted_class, all_predictions)
        
        # Test disposal info
        disposal_info = get_waste_disposal_info(predicted_class)
        assert isinstance(disposal_info, str)
        assert len(disposal_info) > 0
        
        # Test summary creation
        summary = create_prediction_summary(predicted_class, 0.70, all_predictions)
        assert isinstance(summary, str)
        assert predicted_class in summary
        
        print("\n✓ Utility functions work correctly")
        return True
    except Exception as e:
        print(f"✗ Utility test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("WASTE CLASSIFICATION SYSTEM - TEST SUITE")
    print("="*60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Custom Modules", test_custom_modules),
        ("Image Preprocessing", test_preprocessing),
        ("Model Loading", test_model_loading),
        ("Utility Functions", test_utilities),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8s} - {test_name}")
    
    print("-"*60)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
