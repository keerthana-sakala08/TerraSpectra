
import os
import sys
import json
import h5py
import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Allow imports from the TerraSpectra project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.cnn3d_model import TerraSpectra3DCNN


# ==========================================
# TERRASPECTRA - DAY 8
# TRAIN THE 3D-CNN MODEL
# ==========================================

DATASET_PATH = "datasets/processed/cnn_training_dataset.h5"
MODEL_PATH = "models/terraspectra_3dcnn.pth"
METRICS_PATH = "outputs/day8_training_metrics.json"

BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.001
RANDOM_STATE = 42


def load_dataset():

    print("\nLoading Day 6 dataset...")

    with h5py.File(DATASET_PATH, "r") as file:
        X_train = file["X_train"][:]
        y_train = file["y_train"][:]
        X_test = file["X_test"][:]
        y_test = file["y_test"][:]

    print("Training data:", X_train.shape)
    print("Testing data:", X_test.shape)

    if X_train.ndim != 5:
        raise ValueError("Training data must have 5 dimensions.")

    if X_train.shape[1:] != (1, 224, 5, 5):
        raise ValueError("Unexpected training patch dimensions.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training data contains invalid values.")

    if not np.isfinite(X_test).all():
        raise ValueError("Testing data contains invalid values.")

    return X_train, y_train, X_test, y_test


def prepare_data(X_train, y_train, X_test, y_test):

    print("\nPreparing PyTorch tensors...")

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

    dataset = TensorDataset(X_tensor, y_tensor)

    indices = np.arange(len(y_train))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
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


def evaluate(model, loader, criterion, device):

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


def train_model(train_loader, val_loader, device):

    print("\nCreating 3D-CNN model...")

    model = TerraSpectra3DCNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_accuracy = -1.0

    os.makedirs("models", exist_ok=True)

    print("\nStarting training...")

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

            predictions = torch.argmax(outputs, dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = correct / total

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("Best model checkpoint saved.")

    print("\nTraining completed.")

    return best_val_accuracy


def test_model(test_loader, device):

    print("\nEvaluating on separate test dataset...")

    model = TerraSpectra3DCNN().to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model checkpoint not found.")

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    predictions = []
    actual_labels = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)

            outputs = model(inputs)

            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.cpu().numpy())
            actual_labels.extend(labels.numpy())

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    precision = precision_score(
        actual_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        actual_labels,
        predictions,
        labels=[0, 1]
    )

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": matrix.tolist(),
        "test_samples": len(actual_labels),
        "dataset_type": "synthetic",
        "real_world_validation": False
    }

    os.makedirs("outputs", exist_ok=True)

    with open(METRICS_PATH, "w") as file:
        json.dump(results, file, indent=4)

    print("\nTest Results")
    print("------------------------------")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(matrix)

    print("\nModel saved:", MODEL_PATH)
    print("Metrics saved:", METRICS_PATH)

    return results


def main():

    print("\n========== TERRASPECTRA DAY 8 ==========")

    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    X_train, y_train, X_test, y_test = load_dataset()

    train_loader, val_loader, test_loader = prepare_data(
        X_train,
        y_train,
        X_test,
        y_test
    )

    best_val_accuracy = train_model(
        train_loader,
        val_loader,
        device
    )

    results = test_model(
        test_loader,
        device
    )

    print("\nBest validation accuracy:", round(best_val_accuracy, 4))
    print("Final test accuracy:", round(results["accuracy"], 4))

    print("\nDAY 8 COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
