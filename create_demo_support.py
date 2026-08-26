import json
import pandas as pd


TRAIN_FILE = "data/splits/train.csv"

OUTPUT_FILE = "checkpoints/demo_support.json"

K_SHOT = 5

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


metadata = pd.read_csv(
    TRAIN_FILE
)


support_set = {}


for class_name in CLASS_NAMES:

    rows = metadata[
        metadata["dx"] == class_name
    ]

    selected = rows.sample(
        n=K_SHOT,
        random_state=42
    )

    support_set[class_name] = (
        selected["image_id"]
        .tolist()
    )


with open(
    OUTPUT_FILE,
    "w"
) as file:

    json.dump(
        support_set,
        file,
        indent=4
    )


print(
    "\n========== DEMO SUPPORT SET =========="
)

for class_name in CLASS_NAMES:

    print(
        f"{class_name}: "
        f"{len(support_set[class_name])} images"
    )

print(
    "\nSaved:",
    OUTPUT_FILE
)

print(
    "\nDemo support set created successfully!"
)