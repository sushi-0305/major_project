import os
import pandas as pd

from utils.inference import (
    load_model,
    load_image,
    get_embedding,
    calculate_distances
)

import torch


# ==================================================
# CONFIGURATION
# ==================================================

TEST_FILE = "data/splits/test.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

PROTOTYPE_FILE = "checkpoints/prototypes.pt"

CLASS_FILE = "checkpoints/prototype_classes.json"


# ==================================================
# LOAD MODEL
# ==================================================

print("\n========================================")
print("       INFERENCE PIPELINE TEST")
print("========================================")

model = load_model()

device = next(
    model.parameters()
).device


# ==================================================
# LOAD PROTOTYPES
# ==================================================

prototypes = torch.load(
    PROTOTYPE_FILE,
    map_location=device
).to(device)


# ==================================================
# LOAD CLASS NAMES
# ==================================================

import json

with open(
    CLASS_FILE,
    "r"
) as file:

    class_names = json.load(file)


# ==================================================
# LOAD TEST METADATA
# ==================================================

metadata = pd.read_csv(
    TEST_FILE
)


# ==================================================
# SELECT ONE TEST IMAGE PER CLASS
# ==================================================

selected_rows = []

for class_name in class_names:

    rows = metadata[
        metadata["dx"] == class_name
    ]

    if len(rows) == 0:

        continue

    selected_rows.append(
        rows.iloc[0]
    )


# ==================================================
# TEST PREDICTIONS
# ==================================================

correct = 0
total = 0


print(
    "\n========== TEST PREDICTIONS =========="
)


for row in selected_rows:

    image_id = row["image_id"]

    actual_class = row["dx"]

    image_path = None

    for directory in IMAGE_DIRS:

        candidate = os.path.join(
            directory,
            image_id + ".jpg"
        )

        if os.path.exists(candidate):

            image_path = candidate

            break


    if image_path is None:

        print(
            f"\nImage not found: {image_id}"
        )

        continue


    # ----------------------------------------------
    # Load image
    # ----------------------------------------------

    image_tensor = load_image(
        image_path
    )


    # ----------------------------------------------
    # Get embedding
    # ----------------------------------------------

    embedding = get_embedding(
        model,
        image_tensor
    )


    # ----------------------------------------------
    # Calculate distances
    # ----------------------------------------------

    distances = calculate_distances(
        embedding,
        prototypes
    ).squeeze(0)


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    predicted_index = torch.argmin(
        distances
    ).item()

    predicted_class = class_names[
        predicted_index
    ]


    # ----------------------------------------------
    # Result
    # ----------------------------------------------

    is_correct = (
        predicted_class == actual_class
    )

    if is_correct:

        correct += 1

        result = "PASS"

    else:

        result = "MISS"


    total += 1


    print(
        f"\nImage: {image_id}"
    )

    print(
        f"Actual: {actual_class}"
    )

    print(
        f"Predicted: {predicted_class}"
    )

    print(
        f"Result: {result}"
    )


# ==================================================
# SUMMARY
# ==================================================

print(
    "\n========== INFERENCE TEST SUMMARY =========="
)

print(
    "Images tested:",
    total
)

print(
    "Correct:",
    correct
)

if total > 0:

    print(
        f"Accuracy: "
        f"{correct / total * 100:.2f}%"
    )


print(
    "\nInference pipeline test completed successfully!"
)