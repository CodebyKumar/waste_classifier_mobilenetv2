"""Reproducibly train and evaluate an improved waste classifier.

Example:
    uv run python train.py --dataset-dir /path/to/waste_dataset

The dataset directory must contain one folder for each class, for example
``Hazardous/``, ``Non-Recyclable/``, ``Organic/``, and ``Recyclable/``. This
script creates a stratified 70/15/15 train/validation/test split, performs
augmentation only during training, saves the best validation checkpoint, then
fine-tunes the upper MobileNetV2 layers with a lower learning rate.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf
from tensorflow import keras


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
IMAGE_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV2 waste classifier")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, default=Path("models/waste_model_retrained.keras"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=15)
    parser.add_argument("--unfreeze-layers", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def stratified_split(dataset_dir: Path, seed: int) -> tuple[list[str], list[tuple[str, int]], list[tuple[str, int]], list[tuple[str, int]]]:
    class_names = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise ValueError("dataset-dir must contain at least two class folders")

    rng = random.Random(seed)
    train: list[tuple[str, int]] = []
    validation: list[tuple[str, int]] = []
    test: list[tuple[str, int]] = []
    for class_index, class_name in enumerate(class_names):
        paths = image_paths(dataset_dir / class_name)
        if len(paths) < 10:
            raise ValueError(f"{class_name} has only {len(paths)} images; at least 10 are required")
        rng.shuffle(paths)
        validation_end = max(1, int(len(paths) * 0.15))
        test_end = validation_end + max(1, int(len(paths) * 0.15))
        validation.extend((str(path), class_index) for path in paths[:validation_end])
        test.extend((str(path), class_index) for path in paths[validation_end:test_end])
        train.extend((str(path), class_index) for path in paths[test_end:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return class_names, train, validation, test


def make_dataset(
    examples: Iterable[tuple[str, int]],
    batch_size: int,
    training: bool,
    seed: int,
    num_classes: int,
) -> tf.data.Dataset:
    paths, labels = zip(*examples)
    dataset = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
        image = tf.cast(image, tf.float32)
        # MobileNetV2's canonical preprocessing: RGB pixels [0, 255] -> [-1, 1].
        image = (image / 127.5) - 1.0
        return image, tf.one_hot(label, depth=num_classes)

    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    return dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(num_classes: int) -> tuple[keras.Model, keras.Model]:
    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.12),
            keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )
    base = keras.applications.MobileNetV2(
        include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE, 3)
    )
    base.trainable = False

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = augmentation(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    return keras.Model(inputs, outputs, name="waste_classifier"), base


def compile_model(model: keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )


def evaluate_model(model: keras.Model, dataset: tf.data.Dataset, class_names: list[str]) -> dict:
    y_true: list[int] = []
    probabilities: list[np.ndarray] = []
    for images, labels in dataset:
        probabilities.append(model.predict(images, verbose=0))
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
    y_true_array = np.asarray(y_true)
    y_pred = np.argmax(np.concatenate(probabilities), axis=1)
    confusion = tf.math.confusion_matrix(y_true_array, y_pred, num_classes=len(class_names)).numpy()
    support = confusion.sum(axis=1)
    precision = np.divide(np.diag(confusion), confusion.sum(axis=0), out=np.zeros(len(class_names)), where=confusion.sum(axis=0) != 0)
    recall = np.divide(np.diag(confusion), support, out=np.zeros(len(class_names)), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(class_names)), where=(precision + recall) != 0)
    return {
        "accuracy": float(np.mean(y_true_array == y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion.tolist(),
        "per_class": {
            class_name: {
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, class_name in enumerate(class_names)
        },
    }


def main() -> None:
    args = parse_args()
    if not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_dir}")
    if args.output_model.suffix != ".keras":
        raise ValueError("output-model must use the .keras format")
    if args.output_model.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output_model}; use --overwrite to replace it")

    keras.utils.set_random_seed(args.seed)
    class_names, train_examples, validation_examples, test_examples = stratified_split(args.dataset_dir, args.seed)
    print(f"Classes: {class_names}")
    print(f"Split sizes — train: {len(train_examples)}, validation: {len(validation_examples)}, test: {len(test_examples)}")
    train_ds = make_dataset(train_examples, args.batch_size, True, args.seed, len(class_names))
    validation_ds = make_dataset(validation_examples, args.batch_size, False, args.seed, len(class_names))
    test_ds = make_dataset(test_examples, args.batch_size, False, args.seed, len(class_names))

    model, base = build_model(len(class_names))
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        keras.callbacks.ModelCheckpoint(args.output_model, monitor="val_accuracy", mode="max", save_best_only=True),
    ]
    compile_model(model, learning_rate=5e-4)
    model.fit(train_ds, validation_data=validation_ds, epochs=args.head_epochs, callbacks=callbacks)

    base.trainable = True
    for layer in base.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    compile_model(model, learning_rate=1e-5)
    model.fit(train_ds, validation_data=validation_ds, epochs=args.fine_tune_epochs, callbacks=callbacks)

    # Reload the best validation checkpoint before the one-time held-out test evaluation.
    best_model = keras.models.load_model(args.output_model, compile=False)
    metrics = evaluate_model(best_model, test_ds, class_names)
    print(json.dumps(metrics, indent=2))

    # Keep labels beside the new artifact rather than silently replacing the
    # legacy model's labels file.
    labels_path = args.output_model.with_suffix(".labels.txt")
    labels_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    metadata_path = args.output_model.with_suffix(".config.json")
    metadata_path.write_text(
        json.dumps(
            {
                "preprocessing": "mobilenet_v2",
                "confidence_threshold": 0.60,
                "class_names": class_names,
                "split": {"train": len(train_examples), "validation": len(validation_examples), "test": len(test_examples)},
                "test_metrics": metrics,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Saved best model: {args.output_model}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved metrics and model configuration: {metadata_path}")


if __name__ == "__main__":
    main()
