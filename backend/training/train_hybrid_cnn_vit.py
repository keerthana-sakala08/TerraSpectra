
import os
import sys
import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Make the project root available for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.hybrid_cnn_vit import TerraSpectraHybridCNNViT
from backend.models.cnn3d_model import TerraSpectra3DCNN


# ==========================================
# TERRASPECTRA - DAY 14
# HYBRID CNN-VIT TRAINING AND COMPARISON
# ==========================================

DATASET_PATH = PROJECT_ROOT / "datasets/processed/cnn_training_dataset.h5"

HYBRID_MODEL_PATH = PROJECT_ROOT / "models/terraspectra_hybrid_cnn_vit.pth"
CNN_MODEL_PATH = PROJECT_ROOT / "models/terraspectra_3dcnn.pth"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "day14_hybrid_metrics.json"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0005
RANDOM_STATE = 42


def set_seed():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def load_dataset():
    print("\nSTEP 1: Loading hyperspectral dataset")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with h5py.File(DATASET_PATH, "r") as file:
        X_train = file["X_train"][:]
        y_train = file["y_train"][:]
        X_test = file["X_test"][:]
        y_test = file["y_test"][:]

    print("Training data:", X_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing data:", X_test.shape)
    print("Testing labels:", y_test.shape)

    if X_train.ndim != 5:
        raise ValueError("Training data must have 5 dimensions.")

    if X_train.shape[1:] != (1, 224, 5, 5):
        raise ValueError("Unexpected hyperspectral input shape.")

    if X_test.shape[1:] != (1, 224, 5, 5):
        raise ValueError("Unexpected test input shape.")

    if not all(np.isfinite(x).all() for x in (X_train, X_test)):
        raise ValueError("Dataset contains invalid values.")

    if not all(np.isin(y, [0, 1]).all() for y in (y_train, y_test)):
        raise ValueError("Unexpected class labels.")

    return X_train, y_train, X_test, y_test


def prepare_loaders(X_train, y_train, X_test, y_test):
    print("\nSTEP 2: Preparing training and validation loaders")

    X_tensor = torch.from_numpy(
        np.ascontiguousarray(X_train)
    ).float()

    y_tensor = torch.from_numpy(
        np.ascontiguousarray(y_train)
    ).long()

    X_test_tensor = torch.from_numpy(
        np.ascontiguousarray(X_test)
    ).float()

    y_test_tensor = torch.from_numpy(
        np.ascontiguousarray(y_test)
    ).long()

    full_dataset = TensorDataset(X_tensor, y_tensor)

    indices = np.arange(len(y_train))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    generator = torch.Generator()
    generator.manual_seed(RANDOM_STATE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Testing samples:", len(test_dataset))

    return train_loader, val_loader, test_loader


def evaluate_loss(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    average_loss = total_loss / len(loader.dataset)

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    return average_loss, accuracy


def train_hybrid_model(train_loader, val_loader, device):
    print("\nSTEP 3: Creating hybrid CNN-ViT model")

    model = TerraSpectraHybridCNNViT().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    history = []

    HYBRID_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSTEP 4: Starting hybrid model training")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)

            predicted = torch.argmax(outputs, dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate_loss(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_accuracy)
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                HYBRID_MODEL_PATH
            )

            print("Best hybrid checkpoint saved.")

    print("\nHybrid model training completed.")

    return best_val_accuracy, best_epoch, history


def calculate_metrics(actual, predicted):
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1]
    )

    return {
        "accuracy": float(
            accuracy_score(actual, predicted)
        ),
        "precision": float(
            precision_score(
                actual, predicted, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                actual, predicted, zero_division=0
            )
        ),
        "f1_score": float(
            f1_score(
                actual, predicted, zero_division=0
            )
        ),
        "confusion_matrix": matrix.tolist(),
        "test_samples": len(actual)
    }


def predict_model(model, loader, device):
    model.eval()

    predictions = []
    actual_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.numpy())

    return actual_labels, predictions


def evaluate_hybrid_and_cnn(test_loader, device):
    print("\nSTEP 5: Evaluating hybrid CNN-ViT model")

    if not HYBRID_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Hybrid model checkpoint was not saved."
        )

    hybrid_model = TerraSpectraHybridCNNViT().to(device)

    hybrid_model.load_state_dict(
        torch.load(
            HYBRID_MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    actual, hybrid_predictions = predict_model(
        hybrid_model,
        test_loader,
        device
    )

    hybrid_metrics = calculate_metrics(
        actual,
        hybrid_predictions
    )

    print("\nSTEP 6: Evaluating existing Day 8 CNN model")

    if not CNN_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Day 8 CNN checkpoint not found."
        )

    cnn_model = TerraSpectra3DCNN().to(device)

    cnn_model.load_state_dict(
        torch.load(
            CNN_MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    cnn_actual, cnn_predictions = predict_model(
        cnn_model,
        test_loader,
        device
    )

    if actual != cnn_actual:
        raise ValueError("Test labels do not match.")

    cnn_metrics = calculate_metrics(
        cnn_actual,
        cnn_predictions
    )

    print("\nSTEP 7: Comparing model performance")

    print("\nMetric             Day 8 CNN       Day 14 Hybrid")
    print("------------------------------------------------")

    for metric in [
        "accuracy",
        "precision",
        "recall",
        "f1_score"
    ]:
        print(
            f"{metric:<18} "
            f"{cnn_metrics[metric]:.4f}          "
            f"{hybrid_metrics[metric]:.4f}"
        )

    return cnn_metrics, hybrid_metrics


def save_results(
    cnn_metrics,
    hybrid_metrics,
    best_val_accuracy,
    best_epoch,
    history
):
    print("\nSTEP 8: Saving Day 14 results")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "project": "TerraSpectra",
        "day": 14,
        "task": "Hybrid CNN-ViT Training and Evaluation",
        "dataset_type": "synthetic",
        "real_world_validation": False,
        "training_configuration": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "random_state": RANDOM_STATE,
            "validation_split": 0.20
        },
        "hybrid_best_epoch": best_epoch,
        "hybrid_best_validation_accuracy": float(
            best_val_accuracy
        ),
        "training_history": history,
        "cnn_3d_metrics": cnn_metrics,
        "hybrid_cnn_vit_metrics": hybrid_metrics
    }

    with open(METRICS_PATH, "w") as file:
        json.dump(results, file, indent=4)

    print("Hybrid model saved:", HYBRID_MODEL_PATH)
    print("Metrics saved:", METRICS_PATH)


def main():
    print("\n==========================================")
    print(" TERRASPECTRA DAY 14")
    print(" HYBRID CNN-VIT TRAINING")
    print("==========================================")

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    X_train, y_train, X_test, y_test = load_dataset()

    train_loader, val_loader, test_loader = prepare_loaders(
        X_train,
        y_train,
        X_test,
        y_test
    )

    best_val_accuracy, best_epoch, history = train_hybrid_model(
        train_loader,
        val_loader,
        device
    )

    cnn_metrics, hybrid_metrics = evaluate_hybrid_and_cnn(
        test_loader,
        device
    )

    save_results(
        cnn_metrics,
        hybrid_metrics,
        best_val_accuracy,
        best_epoch,
        history
    )

    print("\n==========================================")
    print(" DAY 14 COMPLETED SUCCESSFULLY!")
    print("==========================================")


if __name__ == "__main__":
    main()
