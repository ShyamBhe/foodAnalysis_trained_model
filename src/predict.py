
import sys

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.datasets import Food101

from data import evaluation_transforms, DATA_DIR
from model import build_model, get_device


MODEL_PATH = "models/foodvision_best.pth"


def load_model(device):

    model = build_model()
    model = model.to(device)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def predict_food(image_path):

    device = get_device()

    # Get the 101 Food-101 category names.
    dataset = Food101(
        root=DATA_DIR,
        split="train",
        download=False
    )

    class_names = dataset.classes

    # Load the trained model.
    model = load_model(device)

    # Open and preprocess the input image.
    image = Image.open(image_path).convert("RGB")
    image_tensor = evaluation_transforms(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Generate predictions.
    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)

    # Get the five most likely categories.
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=5,
        dim=1
    )

    print("\n========== FOODVISION AI PREDICTION ==========")
    print("Image:", image_path)
    print("Device:", device)
    print("\nTop 5 predictions:\n")

    for rank, (index, probability) in enumerate(
        zip(
            top_indices[0].cpu().tolist(),
            top_probabilities[0].cpu().tolist()
        ),
        start=1
    ):
        food_name = class_names[index].replace("_", " ")

        print(
            f"{rank}. {food_name.title():25s} "
            f"{probability * 100:.2f}%"
        )

    print("\n========== PREDICTION COMPLETE ==========")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py path/to/food_image.jpg")
        sys.exit(1)

    predict_food(sys.argv[1])
