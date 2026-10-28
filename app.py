from pathlib import Path

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import transforms

from utils.dataset import HAM10000Dataset
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "checkpoints" / "prototypical_skin_model_5shot_best.pth"
TRAIN_FILE = BASE_DIR / "data" / "splits" / "train.csv"

IMAGE_DIRS = [
    BASE_DIR / "data" / "HAM10000_images_part_1",
    BASE_DIR / "data" / "HAM10000_images_part_2",
]

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc",
]

CLASS_LABELS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}

DEVICE = torch.device("cpu")
IMAGE_SIZE = 128
N_SHOT = 5

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def find_image(image_id):
    filename = f"{image_id}.jpg"

    for directory in IMAGE_DIRS:
        path = directory / filename

        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find image: {image_id}.jpg"
    )


def build_support_set():
    """
    Use the same five-shot support images for each class
    from the training dataset.
    """

    dataset = HAM10000Dataset(
        metadata_file=str(TRAIN_FILE),
        image_dirs=[str(path) for path in IMAGE_DIRS],
        transform=transform,
    )

    support_images = []
    support_labels = []

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_rows = dataset.metadata[
            dataset.metadata["dx"] == class_name
        ].head(N_SHOT)

        if len(class_rows) < N_SHOT:
            raise RuntimeError(
                f"Not enough support images for {class_name}"
            )

        for _, row in class_rows.iterrows():

            image_path = find_image(row["image_id"])

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image = transform(image)

            support_images.append(image)
            support_labels.append(class_index)

    support_images = torch.stack(support_images).to(DEVICE)

    support_labels = torch.tensor(
        support_labels,
        dtype=torch.long
    ).to(DEVICE)

    return support_images, support_labels


def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    backbone = FourLayerCNN(
        embedding_dim=64
    )

    model = PrototypicalNetwork(
        backbone
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint)

    model.eval()

    return model


MODEL = None
SUPPORT_IMAGES = None
SUPPORT_LABELS = None


def initialize():

    global MODEL
    global SUPPORT_IMAGES
    global SUPPORT_LABELS

    if MODEL is None:

        print("Loading 5-shot ProtoNet...")

        MODEL = load_model()

        SUPPORT_IMAGES, SUPPORT_LABELS = build_support_set()

        print("Model loaded successfully.")


@app.route("/")
def index():

    return render_template(
        "index.html",
        class_labels=CLASS_LABELS
    )


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:

        return jsonify({
            "error": "Please select an image."
        }), 400

    uploaded = request.files["image"]

    if uploaded.filename == "":

        return jsonify({
            "error": "Please select an image."
        }), 400

    allowed = {
        "jpg",
        "jpeg",
        "png"
    }

    extension = uploaded.filename.rsplit(
        ".",
        1
    )[-1].lower()

    if extension not in allowed:

        return jsonify({
            "error": "Please upload a JPG, JPEG or PNG image."
        }), 400

    try:

        initialize()

        image = Image.open(
            uploaded.stream
        ).convert("RGB")

        query_image = transform(
            image
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            distances, _, _ = MODEL(
                SUPPORT_IMAGES,
                SUPPORT_LABELS,
                query_image
            )

            probabilities = torch.softmax(
                -distances,
                dim=1
            )[0]

        predicted_index = int(
            probabilities.argmax().item()
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        results = []

        for index, class_name in enumerate(CLASS_NAMES):

            results.append({
                "class_name": class_name,
                "label": CLASS_LABELS[class_name],
                "probability": round(
                    float(probabilities[index]) * 100,
                    2
                )
            })

        results.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        return jsonify({

            "prediction":
                CLASS_LABELS[predicted_class],

            "class_name":
                predicted_class,

            "confidence":
                round(
                    float(
                        probabilities[predicted_index]
                    ) * 100,
                    2
                ),

            "results":
                results
        })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    app.run(
        debug=True
    )
