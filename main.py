"""
Main Script for Waste Classification
Orchestrates the entire waste classification pipeline.
"""

import argparse
import sys
from pathlib import Path

# Import custom modules
from src.image_capture import capture_from_camera, load_from_path, save_image
from src.image_preprocessing import preprocess_image
from src.model_inference import WasteClassifier
from src.utils import display_results, print_prediction_results, get_waste_disposal_info


def main():
    """Main function to run the waste classification system."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Waste Classification using MobileNetV2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Capture from camera
  python main.py --camera
  
  # Load from file
  python main.py --image path/to/image.jpg
  
  # Save captured/processed image
  python main.py --camera --save output.jpg
        """
    )
    
    # Add arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--camera', '-c', action='store_true',
                      help='Capture image from camera')
    group.add_argument('--image', '-i', type=str,
                      help='Path to input image file')
    
    parser.add_argument('--save', '-s', type=str,
                      help='Path to save the captured/processed image')
    parser.add_argument('--model', '-m', type=str,
                      default='models/waste_model.h5',
                      help='Path to the trained model (default: models/waste_model.h5)')
    parser.add_argument('--labels', '-l', type=str,
                      default='models/labels.txt',
                      help='Path to the labels file (default: models/labels.txt)')
    parser.add_argument('--no-display', action='store_true',
                      help='Do not display the result image')
    
    args = parser.parse_args()
    
    print("="*60)
    print("WASTE CLASSIFICATION SYSTEM")
    print("="*60)
    
    # Step 1: Capture or load image
    print("\n[1/4] Loading image...")
    if args.camera:
        image = capture_from_camera()
    else:
        image = load_from_path(args.image)
    
    if image is None:
        print("❌ Failed to load image. Exiting.")
        sys.exit(1)
    
    # Save image if requested
    if args.save:
        save_image(image, args.save)
    
    # Step 2: Preprocess image
    print("\n[2/4] Preprocessing image...")
    preprocessed_image = preprocess_image(image)
    print(f"✓ Image preprocessed. Shape: {preprocessed_image.shape}")
    
    # Step 3: Load model and make prediction
    print("\n[3/4] Loading model and making prediction...")
    try:
        classifier = WasteClassifier(
            model_path=args.model,
            labels_path=args.labels
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    # Get prediction with detailed results
    results = classifier.predict_with_confidence(preprocessed_image)
    
    predicted_class = results['predicted_class']
    confidence = results['confidence']
    all_predictions = results['all_predictions']
    
    print("✓ Prediction complete!")
    
    # Step 4: Display results
    print("\n[4/4] Displaying results...")
    
    # Print results to console
    print_prediction_results(predicted_class, all_predictions)
    
    # Show disposal information
    print("\n📋 DISPOSAL INFORMATION:")
    print(get_waste_disposal_info(predicted_class))
    
    # Display image with results (unless --no-display is set)
    if not args.no_display:
        display_results(image, predicted_class, all_predictions)
    
    print("\n✅ Classification complete!")
    print("="*60)


if __name__ == "__main__":
    main()
