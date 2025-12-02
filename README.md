# Waste Classifier MobileNetV2

A waste classification system using MobileNetV2.

## Project Structure

```
.
├── src/                # Source code modules
│   ├── image_capture.py
│   ├── image_preprocessing.py
│   ├── model_inference.py
│   └── utils.py
├── tests/              # Test scripts
│   └── test_system.py
├── notebooks/          # Jupyter notebooks
│   └── workbook.ipynb
├── models/             # Model files
│   ├── waste_model.h5
│   └── labels.txt
├── main.py             # Entry point script
└── requirements.txt    # Dependencies
```

## Usage

### Run the classifier
```bash
python main.py --camera
# or
python main.py --image path/to/image.jpg
```

### Run tests
```bash
python tests/test_system.py
```
