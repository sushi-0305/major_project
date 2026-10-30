import os
import random
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision import transforms

from models.prototypical_networks import PrototypicalNetworks, ConvNet


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SPLIT_DIR = DATA_DIR / "splits"
CHECKPOINT = BASE_DIR / "checkpoints" / "ham10000_prototypical" / "proto_networks_best.pth"

IMAGE_DIRS = [
    DATA_DIR / "HAM10000_images_part_1",
    DATA_DIR / "HAM10000_images_part_2",
]

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
N_SHOT = 5
IMAGE_SIZE = 84

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

DEVICE = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def find_image(image_id):
    filename = f"{image_id}.jpg"
    for directory in IMAGE_DIRS:
        path = directory / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image: {image_id}")


def build_support_set():
    """Use a fixed 5-shot support set from the training split."""
    train_csv = SPLIT_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}")

    df = pd.read_csv(train_csv)

    support_paths = []
    support_labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        rows = df[df["dx"] == class_name].head(N_SHOT)

        if len(rows) < N_SHOT:
            raise RuntimeError(
                f"Not enough training images for class '{class_name}'."
            )

        for _, row in rows.iterrows():
            support_paths.append(find_image(row["image_id"]))
            support_labels.append(class_index)

    images = []
    for path in support_paths:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))

    support_images = torch.stack(images).to(DEVICE)
    support_labels = torch.tensor(support_labels, dtype=torch.long).to(DEVICE)
    return support_images, support_labels


def load_model():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT}"
        )

    backbone = ConvNet(in_channels=3, hidden_dim=64)
    model = PrototypicalNetworks(backbone, hidden_dim=64).to(DEVICE)

    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


MODEL = None
SUPPORT_IMAGES = None
SUPPORT_LABELS = None


def ensure_model():
    global MODEL, SUPPORT_IMAGES, SUPPORT_LABELS
    if MODEL is None:
        MODEL = load_model()
        SUPPORT_IMAGES, SUPPORT_LABELS = build_support_set()


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/")
def index():
    return render_template("index.html", class_labels=CLASS_LABELS)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please select an image."}), 400

    uploaded = request.files["image"]

    if not uploaded.filename:
        return jsonify({"error": "Please select an image."}), 400

    if not allowed_file(uploaded.filename):
        return jsonify({
            "error": "Please upload a JPG, JPEG, or PNG image."
        }), 400

    try:
        ensure_model()

        image = Image.open(uploaded.stream).convert("RGB")
        query = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = MODEL(SUPPORT_IMAGES, SUPPORT_LABELS, query)
            probabilities = torch.softmax(logits, dim=1)[0]

        top_index = int(probabilities.argmax().item())
        confidence = float(probabilities[top_index].item())

        results = []
        for index, class_name in enumerate(CLASS_NAMES):
            results.append({
                "class_name": class_name,
                "label": CLASS_LABELS[class_name],
                "probability": round(float(probabilities[index].item()) * 100, 2),
            })

        results.sort(key=lambda item: item["probability"], reverse=True)

        return jsonify({
            "prediction": CLASS_LABELS[CLASS_NAMES[top_index]],
            "class_name": CLASS_NAMES[top_index],
            "confidence": round(confidence * 100, 2),
            "results": results,
        })

    except Exception as exc:
        return jsonify({
            "error": str(exc)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
