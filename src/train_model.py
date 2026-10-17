"""
train_model.py
--------------
Trains a tomato disease classifier using Transfer Learning (MobileNetV2).

Dataset  : data/plantvillage/split/{train,val,test}/<ClassName>/
Model    : MobileNetV2 (ImageNet weights) + custom classification head
Output   : models/plant_disease_model.keras
           models/class_names.json
           models/training_history.png
           models/evaluation_metrics.json
"""

import json
import os
import sys
import time
from pathlib import Path

# ── Dependency checks ─────────────────────────────────────────────────────────
def check_dependencies():
    missing = []
    for pkg in ["tensorflow", "PIL", "numpy", "matplotlib", "sklearn"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print("        Run:  pip install -r requirements.txt")
        sys.exit(1)

check_dependencies()

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).resolve().parent.parent
SPLIT_DIR  = ROOT_DIR / "data" / "plantvillage" / "split"
MODELS_DIR = ROOT_DIR / "models"

IMG_SIZE    = (224, 224)   # MobileNetV2 native input size
BATCH_SIZE  = 32
EPOCHS      = 50           # EarlyStopping will stop earlier if needed

# Transfer learning schedule
INITIAL_LR        = 1e-3   # Phase 1: train head only
FINE_TUNE_LR      = 1e-5   # Phase 2: fine-tune last layers
FINE_TUNE_LAYERS  = 30     # Number of MobileNetV2 layers to unfreeze in Phase 2

EARLY_STOP_PATIENCE  = 7
REDUCE_LR_PATIENCE   = 3

CLASS_NAMES = [
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
]

# ── Folder validation ─────────────────────────────────────────────────────────

def validate_folders():
    errors = []
    for split in ["train", "val", "test"]:
        split_path = SPLIT_DIR / split
        if not split_path.exists():
            errors.append(f"Missing split folder : {split_path}")
            continue
        for cls in CLASS_NAMES:
            cls_path = split_path / cls
            if not cls_path.exists():
                errors.append(f"Missing class folder : {cls_path}")
            else:
                n = sum(1 for f in cls_path.iterdir() if f.is_file())
                if n == 0:
                    errors.append(f"Empty class folder   : {cls_path}")
    if errors:
        print("\n[ERROR] Dataset validation failed:")
        for e in errors:
            print(f"        {e}")
        print("\n  Run  python src/split_dataset.py  first.")
        sys.exit(1)

# ── Data pipelines ────────────────────────────────────────────────────────────

def build_datasets():
    """Return (train_ds, val_ds, test_ds) as tf.data pipelines."""

    # MobileNetV2 expects pixel values in [-1, 1] — applied in the model head
    # so we load raw uint8 here and normalise inside the model.

    def load_split(name, augment=False):
        ds = keras.utils.image_dataset_from_directory(
            SPLIT_DIR / name,
            class_names=CLASS_NAMES,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=(name == "train"),
            seed=42,
        )
        return ds

    train_ds = load_split("train", augment=True)
    val_ds   = load_split("val")
    test_ds  = load_split("test")

    # Performance optimisation
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds

# ── Model definition ──────────────────────────────────────────────────────────

class MobileNetV2Preprocessor(layers.Layer):
    """
    Serializable Keras layer that scales pixel values from [0, 255] to [-1, 1]
    as expected by MobileNetV2.

    Replaces the previous keras.layers.Lambda usage. Lambda layers store a
    Python lambda as a byte-string and are rejected by Keras safe_mode (default
    since TF 2.12+). A named Layer subclass with get_config() is fully
    serializable — models saved with this layer load cleanly without any
    safe_mode override.
    """
    def call(self, x):
        return tf.keras.applications.mobilenet_v2.preprocess_input(x)

    def get_config(self):
        return super().get_config()


def build_model(num_classes: int) -> keras.Model:
    """
    MobileNetV2 backbone + custom classification head.
    Phase 1: backbone frozen, only head is trained.
    Phase 2: last FINE_TUNE_LAYERS of backbone unfrozen.

    Preprocessing is handled by MobileNetV2Preprocessor (a proper Layer subclass)
    rather than a Lambda, so the saved .keras file is safe to load without
    disabling Keras safe_mode.
    """

    # Augmentation layers (applied only during training via training=True)
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomBrightness(0.2),
        layers.RandomContrast(0.2),
    ], name="augmentation")

    # Preprocessing: scale [0,255] -> [-1,1] using a serializable custom layer.
    # (Replaces the previous layers.Lambda which required safe_mode=False to load.)
    preprocess = MobileNetV2Preprocessor(name="mobilenetv2_preprocess")

    # Backbone
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # frozen in Phase 1

    # Full model
    inputs = keras.Input(shape=(*IMG_SIZE, 3), name="input_image")
    x = augmentation(inputs, training=True)  # augmentation only during .fit()
    x = preprocess(x)
    x = base_model(x, training=False)        # BN layers stay in inference mode
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.4, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="PlantDiseaseClassifier")
    return model, base_model

# ── Training ──────────────────────────────────────────────────────────────────

def train(model, base_model, train_ds, val_ds):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(MODELS_DIR / "best_model.keras")

    callbacks_phase1 = [
        EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=REDUCE_LR_PATIENCE, min_lr=1e-7, verbose=1
        ),
        CSVLogger(str(MODELS_DIR / "training_log_phase1.csv")),
    ]

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 1 — Training classification head (backbone frozen)")
    print("=" * 60)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(show_trainable=True)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_phase1,
        verbose=1,
    )

    # ── Phase 2: Fine-tune top layers of backbone ─────────────────────────────
    print("\n" + "=" * 60)
    print(f"  PHASE 2 — Fine-tuning last {FINE_TUNE_LAYERS} backbone layers")
    print("=" * 60)

    base_model.trainable = True
    # Freeze everything except the last FINE_TUNE_LAYERS
    for layer in base_model.layers[:-FINE_TUNE_LAYERS]:
        layer.trainable = False

    callbacks_phase2 = [
        EarlyStopping(
            monitor="val_accuracy", patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy",
            save_best_only=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=REDUCE_LR_PATIENCE, min_lr=1e-8, verbose=1
        ),
        CSVLogger(str(MODELS_DIR / "training_log_phase2.csv")),
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_phase2,
        verbose=1,
    )

    return history1, history2

# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, test_ds):
    """Evaluate on test set; return loss, accuracy, full classification report."""
    print("\n" + "=" * 60)
    print("  EVALUATING on test set ...")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)

    # Build predictions for detailed report
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
    )

    print("\n  Classification Report:\n")
    print(report_str)

    cm = confusion_matrix(y_true, y_pred)

    return test_loss, test_acc, report, report_str, cm

# ── Plotting ──────────────────────────────────────────────────────────────────

def save_training_plots(history1, history2, cm):
    """Save accuracy/loss curves and confusion matrix to models/."""

    # Combine history from both phases
    def concat(key):
        h1 = history1.history.get(key, [])
        h2 = history2.history.get(key, [])
        return h1 + h2

    acc      = concat("accuracy")
    val_acc  = concat("val_accuracy")
    loss     = concat("loss")
    val_loss = concat("val_loss")
    epochs   = range(1, len(acc) + 1)
    phase2_start = len(history1.history["accuracy"]) + 1

    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig)

    # Accuracy plot
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(epochs, acc,     label="Train Acc",  linewidth=2)
    ax1.plot(epochs, val_acc, label="Val Acc",    linewidth=2, linestyle="--")
    ax1.axvline(x=phase2_start, color="gray", linestyle=":", label="Fine-tune start")
    ax1.set_title("Accuracy", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Loss plot
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(epochs, loss,     label="Train Loss", linewidth=2)
    ax2.plot(epochs, val_loss, label="Val Loss",   linewidth=2, linestyle="--")
    ax2.axvline(x=phase2_start, color="gray", linestyle=":", label="Fine-tune start")
    ax2.set_title("Loss", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    # Confusion matrix
    ax3 = fig.add_subplot(gs[2])
    im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax3)
    tick_marks = np.arange(len(CLASS_NAMES))
    short_names = [c.replace("Tomato_", "") for c in CLASS_NAMES]
    ax3.set_xticks(tick_marks); ax3.set_xticklabels(short_names, rotation=30, ha="right", fontsize=8)
    ax3.set_yticks(tick_marks); ax3.set_yticklabels(short_names, fontsize=8)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax3.set_title("Confusion Matrix (Test)", fontsize=13)
    ax3.set_ylabel("True label"); ax3.set_xlabel("Predicted label")

    plt.suptitle("PlantVillage — MobileNetV2 Training Results", fontsize=15, y=1.02)
    plt.tight_layout()
    plot_path = MODELS_DIR / "training_history.png"
    plt.savefig(plot_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\n  Training plots saved → {plot_path}")

# ── Save artefacts ────────────────────────────────────────────────────────────

def save_artefacts(model, test_loss, test_acc, report, report_str):
    """Save final model, class names, and evaluation metrics."""

    # Final model
    model_path = MODELS_DIR / "plant_disease_model.keras"
    model.save(model_path)
    print(f"  Model saved        → {model_path}")

    # Class names (needed by the Streamlit app)
    names_path = MODELS_DIR / "class_names.json"
    with open(names_path, "w") as f:
        json.dump(CLASS_NAMES, f, indent=2)
    print(f"  Class names saved  → {names_path}")

    # Evaluation metrics
    metrics = {
        "test_loss":     round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "classification_report": report,
    }
    metrics_path = MODELS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved      → {metrics_path}")

    # Human-readable text report
    txt_path = MODELS_DIR / "classification_report.txt"
    with open(txt_path, "w") as f:
        f.write(f"Test Loss     : {test_loss:.4f}\n")
        f.write(f"Test Accuracy : {test_acc:.4f}\n\n")
        f.write(report_str)
    print(f"  Text report saved  → {txt_path}")

# ── Memory / GPU info ─────────────────────────────────────────────────────────

def print_environment():
    print("\n" + "=" * 60)
    print("  Environment")
    print("=" * 60)
    print(f"  TensorFlow version : {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for g in gpus:
            print(f"  GPU detected       : {g.name}")
        # Allow memory growth to avoid OOM on smaller GPUs
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print("  GPU memory growth  : enabled")
        except RuntimeError as e:
            print(f"  [WARN] {e}")
    else:
        print("  GPU detected       : None — training on CPU (slower)")
        print("  Tip: Training on CPU may take 30-90 minutes for this dataset.")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    print("\n" + "=" * 60)
    print("  PlantDiseaseAI — Model Training")
    print("=" * 60)

    print_environment()

    # Validate dataset
    print("\n  Validating dataset folders ...")
    validate_folders()
    print("  All dataset folders OK.")

    # Build data pipelines
    print("\n  Loading dataset ...")
    train_ds, val_ds, test_ds = build_datasets()

    # Count samples
    n_train = sum(1 for _ in train_ds.unbatch())
    n_val   = sum(1 for _ in val_ds.unbatch())
    n_test  = sum(1 for _ in test_ds.unbatch())
    print(f"  Train : {n_train} images")
    print(f"  Val   : {n_val} images")
    print(f"  Test  : {n_test} images")
    print(f"  Classes : {CLASS_NAMES}")

    # Build model
    print("\n  Building model ...")
    model, base_model = build_model(num_classes=len(CLASS_NAMES))

    # Train
    history1, history2 = train(model, base_model, train_ds, val_ds)

    # Evaluate
    test_loss, test_acc, report, report_str, cm = evaluate(model, test_ds)

    # Save everything
    print("\n" + "=" * 60)
    print("  Saving artefacts ...")
    print("=" * 60)
    save_artefacts(model, test_loss, test_acc, report, report_str)
    save_training_plots(history1, history2, cm)

    elapsed = time.time() - start_time
    mins, secs = divmod(int(elapsed), 60)

    print("\n" + "=" * 60)
    print(f"  Training complete in {mins}m {secs}s")
    print(f"  Final test accuracy : {test_acc * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("\n[ERROR] Out of memory!")
        print("  Try reducing BATCH_SIZE (currently 32) to 16 or 8 in train_model.py")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[ABORTED] Training interrupted by user.")
        sys.exit(0)
