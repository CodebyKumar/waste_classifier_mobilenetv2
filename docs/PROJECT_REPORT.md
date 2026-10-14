# Smart Waste Classifier — Project Report

## Purpose

This project classifies a submitted waste image into one of four categories:
**Hazardous**, **Non-Recyclable**, **Organic**, or **Recyclable**. It then shows
the probability for each category and general disposal guidance. It is a local
decision-support demo, not an authority on local waste regulations or a safe
replacement for manual review of potentially hazardous items.

## What is currently implemented

| Area | Implementation |
| --- | --- |
| Model artifact | `models/waste_model.h5`, a 9.2 MB Keras MobileNetV2-based classifier with four outputs. |
| Label mapping | `models/labels.txt` maps output indices 0–3 to Hazardous, Non-Recyclable, Organic, and Recyclable. |
| Preprocessing | OpenCV BGR input is resized to 224×224, converted to RGB, divided by 255, and given a batch dimension. This matches the legacy notebook. |
| Prediction layer | `src/model_inference.py` loads the model/labels, validates their output-size contract, returns probabilities, the top class, and a review flag. |
| Input methods | `src/image_capture.py` supports local images and interactive webcam capture. |
| Interfaces | `main.py` supplies a CLI; `app.py` provides a Gradio upload interface. |
| User output | Both interfaces show class probabilities; the web UI and CLI provide category-specific disposal notes. |
| Checks | `tests/test_system.py` performs basic integration-style checks for imports, preprocessing, model loading, prediction structure, and utilities. |

## Existing training approach

The original notebook (`notebooks/workbook.ipynb`) downloaded the Kaggle dataset
`phenomsg/waste-classification`, found 2,884 images, and used a directory-based
80/20 train/validation split:

1. It applied rescaling, rotation, zoom, and horizontal flips with
   `ImageDataGenerator`.
2. It used ImageNet-pretrained MobileNetV2 without its original classification
   head.
3. It froze the entire MobileNetV2 base and trained global average pooling,
   dropout (0.3), and a four-class softmax layer for 12 epochs.
4. It exported an HDF5 model and a labels text file.

The notebook records approximately **73.8% final training accuracy**, **53.0%
best validation accuracy**, and **51.8% final validation accuracy**. These are
validation figures from the notebook, not an independent test result. They
should not be presented as production accuracy or placed on a resume as a final
metric.

## Limitations found

The legacy model is usable as a demo, but its evaluation is not sufficient for
a reliability claim:

- The original validation generator applied random augmentation, so validation
  scores can vary and do not represent untouched data.
- There is no independent held-out test set.
- The gap between recorded training and validation accuracy suggests overfitting.
- Training was limited to a frozen backbone; no low-learning-rate fine-tuning
  stage was used.
- Early stopping, learning-rate reduction, and model checkpointing were absent.
- No class balance report, precision/recall/F1, or confusion matrix was saved.
- Softmax values are relative model scores, not proof that a disposal decision
  is correct. In particular, a low-confidence hazardous-waste prediction should
  be checked manually.
- Disposal rules differ by location; the provided advice remains general.

## Improvements made in this revision

### Inference safety and correctness

- The model wrapper now checks that the number of labels equals the model’s
  output width before predictions are accepted.
- It validates incoming tensors and loads inference models with
  `compile=False`, avoiding unnecessary training-state deserialization.
- A confidence-review signal is produced using a configurable 60% threshold.
  The CLI and web report warn the user to verify low-confidence results before
  relying on disposal guidance.
- Preprocessing now validates images, supports the legacy `[0, 1]` convention,
  and supports MobileNetV2’s canonical `[-1, 1]` convention for newly trained
  models. A model-sidecar configuration keeps the inference transformation
  coupled to the training transformation.

### Reproducible retraining

`train.py` is an added replacement training entry point. It does not overwrite
the legacy `.h5` model by default. It:

1. Creates deterministic, class-stratified 70/15/15 train/validation/test
   partitions using a specified seed.
2. Applies augmentation only to the training path, inside the Keras model.
3. Uses MobileNetV2’s canonical preprocessing for the new model.
4. Trains the new classification head with an ImageNet-pretrained frozen base.
5. Fine-tunes only the last 40 base layers at a lower learning rate.
6. Uses early stopping, learning-rate reduction, and best-validation checkpointing.
7. Evaluates only once on the held-out test set and writes accuracy, macro
   precision/recall/F1, a confusion matrix, and per-class metrics to the model
   sidecar JSON.

## How to run

Install the locked dependencies, then run the checks:

```bash
uv sync
uv run python tests/test_system.py
uv run python main.py --image test_images/image.png --no-display
uv run python app.py
```

To train a new model, obtain a dataset organised by class folders and run:

```bash
uv run python train.py --dataset-dir /path/to/waste_dataset
```

This produces `models/waste_model_retrained.keras`, its matching
`models/waste_model_retrained.config.json`, and
`models/waste_model_retrained.labels.txt`. Use the new artifact with the CLI
after reviewing the recorded held-out metrics:

```bash
uv run python main.py --image test_images/image.png --no-display \
  --model models/waste_model_retrained.keras \
  --labels models/waste_model_retrained.labels.txt
```

For the Gradio interface, update both `MODEL_PATH` and `LABELS_PATH` in
`app.py` only after the new model passes evaluation. The metadata sidecar is
discovered automatically when it is beside the selected model.

## Validation performed in this environment

Static compilation of the Python source was performed after the changes.
Runtime model and integration tests were attempted, but could not be completed
here because the existing virtual environment has no installed dependencies and
PyPI DNS access was unavailable, including after a dependency-sync attempt.
This is an environment limitation, not evidence that the model passed runtime
validation. Run the commands above in a connected environment before publishing
metrics or replacing the legacy artifact.

## Recommended next work

1. Run `train.py`, inspect the per-class test metrics and confusion matrix, and
   only then choose a model for the application.
2. Add real, labelled test images—especially difficult lighting, backgrounds,
   multiple items, and non-waste images—and test expected predictions.
3. Tune the review threshold from validation data and evaluate calibration;
   avoid presenting a softmax score as guaranteed correctness.
4. Add local disposal rules by country/city or clearly link to the relevant
   municipal waste authority.
5. Pin a tested Python/TensorFlow combination. The current project requires
   Python 3.13, which can reduce compatibility with some TensorFlow versions.

