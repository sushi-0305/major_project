import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def calculate_metrics(
    true_labels,
    predicted_labels
):
    """
    Calculate classification metrics.
    """

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    precision = precision_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }


def evaluate_model(
    model,
    dataloader,
    device="cpu"
):
    """
    Evaluate a standard classification model.
    """

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            true_labels.extend(
                labels.cpu().numpy()
            )

            predicted_labels.extend(
                predictions.cpu().numpy()
            )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels
    )

    return metrics


def print_metrics(metrics):
    """
    Print evaluation results.
    """

    print("\n========== EVALUATION RESULTS ==========")

    print(
        f"Accuracy  : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision : "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1_score']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])

    print("=========================================\n")