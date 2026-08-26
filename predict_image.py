import sys
import torch
import json
import os

from utils.inference import (
    load_model,
    load_image,
    get_embedding,
    calculate_distances
)


# ==================================================
# CONFIGURATION
# ==================================================

PROTOTYPE_FILE = "checkpoints/prototypes.pt"

CLASS_FILE = "checkpoints/prototype_classes.json"


# ==================================================
# CHECK ARGUMENT
# ==================================================

if len(sys.argv) != 2:

    print(
        "\nUsage:"
    )

    print(
        "python predict_image.py <image_path>"
    )

    sys.exit(1)


image_path = sys.argv[1]


# ==================================================
# CHECK IMAGE
# ==================================================

if not os.path.exists(image_path):

    print(
        f"\nImage not found: {image_path}"
    )

    sys.exit(1)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "\n========================================"
)

print(
    "       FEW-SHOT SKIN DISEASE MODEL"
)

print(
    "========================================"
)

print(
    "Using device:",
    device
)

print(
    "Image:",
    image_path
)


# ==================================================
# LOAD MODEL
# ==================================================

model = load_model()


# ==================================================
# LOAD PROTOTYPES
# ==================================================

prototypes = torch.load(
    PROTOTYPE_FILE,
    map_location=device
)

prototypes = prototypes.to(device)


# ==================================================
# LOAD CLASS NAMES
# ==================================================

with open(
    CLASS_FILE,
    "r"
) as file:

    class_names = json.load(
        file
    )


# ==================================================
# LOAD IMAGE
# ==================================================

image_tensor = load_image(
    image_path
)


# ==================================================
# GET EMBEDDING
# ==================================================

embedding = get_embedding(
    model,
    image_tensor
)


# ==================================================
# CALCULATE DISTANCES
# ==================================================

distances = calculate_distances(
    embedding,
    prototypes
)


distances = distances.squeeze(0)


# ==================================================
# PREDICTION
# ==================================================

predicted_index = torch.argmin(
    distances
).item()

predicted_class = class_names[
    predicted_index
]


# ==================================================
# DISTANCE-BASED SCORE
# ==================================================

# Convert distances into relative scores.
scores = torch.softmax(
    -distances,
    dim=0
)

predicted_score = scores[
    predicted_index
].item()


# ==================================================
# DISPLAY RESULT
# ==================================================

print(
    "\n========== PREDICTION =========="
)

print(
    "Predicted class:",
    predicted_class
)

print(
    f"Relative prototype score: "
    f"{predicted_score * 100:.2f}%"
)


print(
    "\n========== CLASS SCORES =========="
)

for index, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:6s}: "
        f"{scores[index].item() * 100:.2f}%"
    )


print(
    "\n========== DISTANCES =========="
)

for index, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:6s}: "
        f"{distances[index].item():.4f}"
    )


print(
    "\n========================================"
)

print(
    "Prediction completed successfully!"
)

print(
    "For research/demo purposes only."
)

print(
    "This model is not a medical diagnostic tool."
)

print(
    "========================================"
)