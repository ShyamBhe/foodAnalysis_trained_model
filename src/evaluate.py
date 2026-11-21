
import os

import torch
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data import get_data_loaders
from model import build_model, get_device


MODEL_PATH = "models/foodvision_best.pth"
RESULTS_DIR = "results"


def main():

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = get_device()

    print("\n========== FOODVISION AI EVALUATION ==========")
    print("Selected device:", device)

    # Load the official test dataset.
    _, _, test_loader = get_data_loaders()

    class_names = test_loader.dataset.classes

    # Initialize model architecture.
    model = build_model()
    model = model.to(device)

    # Load the best trained checkpoint.
    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Best training epoch:", checkpoint["epoch"])
    print("Best validation accuracy:",
          f"{checkpoint['validation_accuracy']:.2f}%")

    model.eval()

    all_predictions = []
    all_labels = []

    # Run predictions without calculating gradients.
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Calculate overall accuracy.
    accuracy = accuracy_score(all_labels, all_predictions)

    print("\n========== TEST RESULTS ==========")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Calculate metrics for all 101 categories.
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_path = os.path.join(
        RESULTS_DIR,
        "classification_report.csv"
    )

    report_df.to_csv(report_path)

    # Print overall metrics.
    print("\n========== OVERALL METRICS ==========")

    for metric in ["macro avg", "weighted avg"]:
        print(f"\n{metric.upper()}")
        print(f"Precision: {report[metric]['precision']:.4f}")
        print(f"Recall:    {report[metric]['recall']:.4f}")
        print(f"F1-score:  {report[metric]['f1-score']:.4f}")

    # Create confusion matrix.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names)))
    )

    plt.figure(figsize=(20, 18))

    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=90,
        fontsize=6
    )

    plt.yticks(
        range(len(class_names)),
        class_names,
        fontsize=6
    )

    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.title("FoodVision AI - Food-101 Confusion Matrix")

    plt.tight_layout()

    confusion_path = os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(confusion_path, dpi=200)
    plt.close()

    print("\nEvaluation files saved:")
    print(report_path)
    print(confusion_path)

    print("\n========== EVALUATION COMPLETE ==========")


if __name__ == "__main__":
    main()
