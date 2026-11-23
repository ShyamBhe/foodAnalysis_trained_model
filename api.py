
import io
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.data import evaluation_transforms
from src.model import build_model, get_device


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "foodvision_finetuned_best.pth"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

DEVICE = get_device()

app = FastAPI(
    title="FoodVision AI API",
    description="AI-powered food image classification using MobileNetV3-Small.",
    version="1.0.0"
)


# --------------------------------------------------
# Load model and class names
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(f"Class names not found: {CLASS_NAMES_PATH}")

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

model = build_model()
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

print("FoodVision AI model loaded successfully.")
print("Device:", DEVICE)
print("Number of food categories:", len(class_names))


# --------------------------------------------------
# API endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to FoodVision AI API",
        "model": "MobileNetV3-Small",
        "categories": len(class_names),
        "device": str(DEVICE),
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(DEVICE)
    }


@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):

    # Validate file type.
    if file.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp"
    }:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPEG, PNG or WEBP image."
        )

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded image is empty."
            )

        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Image must be smaller than 10 MB."
            )

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        )

    except OSError:
        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    # Apply the same preprocessing used during evaluation.
    image_tensor = evaluation_transforms(image).unsqueeze(0)
    image_tensor = image_tensor.to(DEVICE)

    # Run model inference.
    try:
        with torch.inference_mode():
            outputs = model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=5,
            dim=1
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )

    predictions = []

    for index, probability in zip(
        top_indices[0].cpu().tolist(),
        top_probabilities[0].cpu().tolist()
    ):
        predictions.append({
            "food": class_names[index].replace("_", " ").title(),
            "confidence": round(probability * 100, 2)
        })

    return {
        "filename": file.filename,
        "predictions": predictions,
        "device": str(DEVICE)
    }
