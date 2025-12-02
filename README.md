# 🌍 Smart Waste Classifier

An AI-powered waste classification system using MobileNetV2 deep learning model to identify and categorize waste into four categories: **Recyclable**, **Organic**, **Hazardous**, and **Non-Recyclable**. Get instant disposal instructions and helpful tips!

## ✨ Features

- 🎯 **Accurate Classification**: Uses MobileNetV2 neural network for high-accuracy waste identification
- 🌐 **Web Interface**: User-friendly Gradio web app for easy image upload and classification
- 💻 **Command Line Interface**: Flexible CLI for camera capture or file-based classification
- 📊 **Detailed Results**: Visual probability bars and confidence scores for all categories
- ♻️ **Disposal Guidance**: Specific disposal instructions and helpful tips for each waste type
- 🔍 **Real-time Predictions**: Instant classification with detailed breakdown

## 🗂️ Project Structure

```
waste_classifier_mobilenetv2/
├── src/                    # Source code modules
│   ├── __init__.py
│   ├── image_capture.py    # Camera and image loading utilities
│   ├── image_preprocessing.py  # Image preprocessing pipeline
│   ├── model_inference.py  # Model loading and prediction
│   └── utils.py            # Display and formatting utilities
├── tests/                  # Test scripts
│   └── test_system.py      # System integration tests
├── notebooks/              # Jupyter notebooks
│   └── workbook.ipynb      # Training and experimentation notebook
├── models/                 # Model files
│   ├── waste_model.h5      # Trained MobileNetV2 model
│   └── labels.txt          # Class labels
├── app.py                  # Gradio web application
├── main.py                 # Command-line interface
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd waste_classifier_mobilenetv2
```

2. **Install dependencies**

Using uv (recommended):
```bash
uv sync
# or
uv add -r requirements.txt
```

Using pip (alternative):
```bash
pip install -r requirements.txt
```

## 📖 Usage

### Web Application (Recommended)

Launch the Gradio web interface:
```bash
uv run python app.py
# or if using pip
python app.py
```

Then open your browser to `http://127.0.0.1:7860` and upload an image to classify.

### Command Line Interface

**Capture from camera:**
```bash
uv run python main.py --camera
```

**Classify an image file:**
```bash
uv run python main.py --image path/to/image.jpg
```

**Save the processed image:**
```bash
uv run python main.py --camera --save output.jpg
```

**Custom model paths:**
```bash
uv run python main.py --image waste.jpg --model models/waste_model.h5 --labels models/labels.txt
```

**Disable image display:**
```bash
uv run python main.py --image waste.jpg --no-display
```

### Run Tests

```bash
uv run python tests/test_system.py
```

## 🎯 Waste Categories

### ♻️ Recyclable
- **Examples**: Paper, cardboard, glass bottles, aluminum cans, plastic bottles
- **Disposal**: Place in recycling bin after cleaning and drying

### 🌱 Organic
- **Examples**: Food scraps, yard waste, paper towels, coffee grounds
- **Disposal**: Compost at home or use green waste bin

### ⚠️ Hazardous
- **Examples**: Batteries, chemicals, paint, electronics, fluorescent bulbs
- **Disposal**: Take to designated hazardous waste collection center

### 🗑️ Non-Recyclable
- **Examples**: Styrofoam, plastic bags, chip bags, contaminated materials
- **Disposal**: General waste bin

## 🛠️ Technical Details

- **Model**: MobileNetV2 (pre-trained on ImageNet, fine-tuned for waste classification)
- **Input Size**: 224x224x3 RGB images
- **Framework**: TensorFlow/Keras
- **Web Framework**: Gradio
- **Image Processing**: OpenCV

## 📊 Model Performance

The model classifies waste into 4 categories with confidence scores. Check the detailed report in the web interface for probability distributions across all classes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

[Add your license here]

## 👨‍💻 Author

[Add your name/contact here]

## 🙏 Acknowledgments

- MobileNetV2 architecture by Google
- Dataset: [Add dataset source if applicable]
