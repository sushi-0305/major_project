import os
import sys
import random

import torch
from flask import Flask, request, jsonify

import json
from PIL import Image

# Allow app/ to import project utilities
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from utils.model import FourLayerCNN
from utils.data_augmentation import get_test_transform


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_FILE = (
    "prototypical_skin_model_5shot.pth"
)

IMAGE_SIZE = 128
EMBEDDING_DIM = 64

UPLOAD_FOLDER = "app/uploads"

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# ==================================================
# FLASK APP
# ==================================================

app = Flask(__name__)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# LOAD MODEL
# ==================================================

model = FourLayerCNN(
    embedding_dim=EMBEDDING_DIM
).to(device)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)


backbone_state = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):

        new_key = key.replace(
            "backbone.",
            "",
            1
        )

        backbone_state[new_key] = value


model.load_state_dict(
    backbone_state
)

model.eval()


# ==================================================
# TRANSFORM
# ==================================================

transform = get_test_transform(
    image_size=IMAGE_SIZE
)


# ==================================================
# HOME ROUTE
# ==================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return jsonify({
        "message":
            "Few-Shot Skin Disease Classification API",
        "status":
            "running",
        "model":
            "4-Layer CNN + Prototypical Network",
        "setting":
            "5-way 5-shot"
    })


# ==================================================
# HEALTH CHECK
# ==================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "healthy",
        "device": str(device),
        "model_loaded": True
    })


# ==================================================
# IMAGE PREDICTION
# ==================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # ==================================================
    # CHECK UPLOADED IMAGE
    # ==================================================

    if "image" not in request.files:

        return jsonify({
            "success": False,
            "error":
                "No image uploaded. "
                "Use field name 'image'."
        }), 400


    file = request.files["image"]


    if file.filename == "":

        return jsonify({
            "success": False,
            "error": "No image selected."
        }), 400


    # ==================================================
    # SAVE TEMPORARY IMAGE
    # ==================================================

    filename = file.filename

    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    file.save(filepath)


    # ==================================================
    # LOAD QUERY IMAGE
    # ==================================================

    try:

        image = Image.open(
            filepath
        ).convert("RGB")

    except Exception:

        return jsonify({
            "success": False,
            "error": "Invalid image file."
        }), 400


    # ==================================================
    # TRANSFORM QUERY IMAGE
    # ==================================================

    query_tensor = transform(
        image
    )

    query_tensor = query_tensor.unsqueeze(
        0
    ).to(device)


    # ==================================================
    # LOAD FIXED 5-SHOT SUPPORT SET
    # ==================================================

    SUPPORT_FILE = (
        "checkpoints/demo_support.json"
    )

    with open(
        SUPPORT_FILE,
        "r"
    ) as support_file:

        support_set = json.load(
            support_file
        )


    # ==================================================
    # CREATE SUPPORT EMBEDDINGS
    # ==================================================

    support_embeddings = []

    support_class_names = []


    with torch.no_grad():

        for class_name in CLASS_NAMES:

            class_images = []

            for image_id in support_set[
                class_name
            ]:

                support_path = None

                for directory in [
                    "data/HAM10000_images_part_1",
                    "data/HAM10000_images_part_2"
                ]:

                    candidate = os.path.join(
                        directory,
                        image_id + ".jpg"
                    )

                    if os.path.exists(
                        candidate
                    ):

                        support_path = candidate

                        break


                if support_path is None:

                    continue


                support_image = Image.open(
                    support_path
                ).convert("RGB")


                support_image = transform(
                    support_image
                )


                class_images.append(
                    support_image
                )


            # ------------------------------------------
            # Make sure all 5 support images exist
            # ------------------------------------------

            if len(class_images) == 0:

                continue


            class_images = torch.stack(
                class_images
            ).to(device)


            # ------------------------------------------
            # Generate embeddings
            # ------------------------------------------

            embeddings = model(
                class_images
            )


            # ------------------------------------------
            # Calculate class prototype
            # ------------------------------------------

            prototype = embeddings.mean(
                dim=0
            )


            support_embeddings.append(
                prototype
            )

            support_class_names.append(
                class_name
            )


        # ==================================================
        # QUERY EMBEDDING
        # ==================================================

        query_embedding = model(
            query_tensor
        )


        # ==================================================
        # CREATE PROTOTYPE TENSOR
        # ==================================================

        prototypes = torch.stack(
            support_embeddings
        )


        # ==================================================
        # CALCULATE DISTANCES
        # ==================================================

        distances = torch.cdist(
            query_embedding,
            prototypes
        ).squeeze(0)


    # ==================================================
    # SORT PREDICTIONS
    # ==================================================

    sorted_indices = torch.argsort(
        distances
    )


    predicted_index = sorted_indices[
        0
    ].item()


    predicted_class = (
        support_class_names[
            predicted_index
        ]
    )


    # ==================================================
    # CALCULATE RELATIVE SIMILARITY
    # ==================================================

    scores = torch.softmax(
        -distances,
        dim=0
    )


    predicted_score = scores[
        predicted_index
    ].item()


    # ==================================================
    # DISTANCE INFORMATION
    # ==================================================

    distance_results = {}

    for index, class_name in enumerate(
        support_class_names
    ):

        distance_results[
            class_name
        ] = round(
            distances[index].item(),
            4
        )


    # ==================================================
    # TOP 3 PREDICTIONS
    # ==================================================

    top_predictions = []

    for index in sorted_indices[
        :3
    ]:

        index = index.item()

        top_predictions.append({

            "class":
                support_class_names[index],

            "similarity":
                round(
                    scores[index].item() * 100,
                    2
                ),

            "distance":
                round(
                    distances[index].item(),
                    4
                )
        })


    # ==================================================
    # RESPONSE
    # ==================================================

    return jsonify({

        "success": True,

        "prediction":
            predicted_class,

        "prototype_similarity":
            round(
                predicted_score * 100,
                2
            ),

        "distances":
            distance_results,

        "top_predictions":
            top_predictions,

        "classes":
            support_class_names,

        "support_shots":
            5,

        "embedding_dimension":
            int(
                query_embedding.shape[1]
            ),

        "model":
            "4-Layer CNN + Prototypical Network",

        "setting":
            "5-way 5-shot",

        "note":
            "Prototype similarity is not "
            "a calibrated medical probability. "
            "This system is for research and "
            "demonstration purposes only."
    })

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "   FEW-SHOT SKIN DISEASE API"
    )

    print(
        "========================================"
    )

    print(
        "Device:",
        device
    )

    print(
        "Model:",
        "4-Layer CNN + Prototypical Network"
    )

    print(
        "Setting:",
        "5-way 5-shot"
    )

    print(
        "\nServer starting..."
    )

    app.run(
    host="127.0.0.1",
    port=5000,
    debug=False,
    use_reloader=False,
    threaded=True
    )