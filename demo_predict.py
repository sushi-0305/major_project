import os
import random
import json
import torch

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork


# ==================================================
# CONFIGURATION
# ==================================================

TRAIN_FILE = "data/splits/train.csv"
TEST_FILE = "data/splits/test.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

MODEL_FILE = "prototypical_skin_model_5shot.pth"

IMAGE_SIZE = 128
EMBEDDING_DIM = 64

N_WAY = 5
K_SHOT = 5

SEED = 42

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
# REPRODUCIBILITY
# ==================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\n========================================")
print("       5-WAY 5-SHOT PROTONET DEMO")
print("========================================")

print(
    "Using device:",
    device
)


# ==================================================
# LOAD TRAINING DATASET
# ==================================================

train_dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(
        image_size=IMAGE_SIZE
    )
)

print(
    "\nTraining images:",
    len(train_dataset)
)


# ==================================================
# LOAD TEST DATASET
# ==================================================

test_dataset = HAM10000Dataset(
    metadata_file=TEST_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(
        image_size=IMAGE_SIZE
    )
)

print(
    "Test images:",
    len(test_dataset)
)


# ==================================================
# LOAD MODEL
# ==================================================

backbone = FourLayerCNN(
    embedding_dim=EMBEDDING_DIM
)

model = PrototypicalNetwork(
    backbone
).to(device)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

model.load_state_dict(
    checkpoint
)

model.eval()

print(
    "\nTrained ProtoNet loaded successfully!"
)


# ==================================================
# SELECT FIVE CLASSES
# ==================================================

selected_classes = random.sample(
    CLASS_NAMES,
    N_WAY
)

print(
    "\n========== SELECTED CLASSES =========="
)

for class_name in selected_classes:

    print(
        "-",
        class_name
    )


# ==================================================
# CREATE SUPPORT SET
# ==================================================

support_images = []
support_labels = []

support_records = []


for new_label, class_name in enumerate(
    selected_classes
):

    class_rows = train_dataset.metadata[
        train_dataset.metadata["dx"] == class_name
    ]

    selected_rows = class_rows.sample(
        n=K_SHOT,
        random_state=SEED + new_label
    )

    for _, row in selected_rows.iterrows():

        image_id = row["image_id"]

        image_index = train_dataset.metadata.index[
            train_dataset.metadata["image_id"]
            == image_id
        ][0]

        image, original_label = train_dataset[
            image_index
        ]

        support_images.append(
            image
        )

        # ProtoNet episode labels must be
        # 0, 1, 2, 3, 4
        support_labels.append(
            new_label
        )

        support_records.append(
            {
                "image_id": image_id,
                "class": class_name
            }
        )


support_images = torch.stack(
    support_images
).to(device)

support_labels = torch.tensor(
    support_labels,
    dtype=torch.long
).to(device)


# ==================================================
# CREATE PROTOTYPES
# ==================================================

with torch.no_grad():

    support_embeddings = model.backbone(
        support_images
    )

    prototypes = []

    for class_index in range(
        N_WAY
    ):

        class_embeddings = (
            support_embeddings[
                support_labels == class_index
            ]
        )

        prototype = class_embeddings.mean(
            dim=0
        )

        prototypes.append(
            prototype
        )

    prototypes = torch.stack(
        prototypes
    )


# ==================================================
# DISPLAY SUPPORT SET
# ==================================================

print(
    "\n========== SUPPORT SET =========="
)

for class_name in selected_classes:

    count = sum(
        1
        for record in support_records
        if record["class"] == class_name
    )

    print(
        f"{class_name}: "
        f"{count} images"
    )


# ==================================================
# SELECT TEST IMAGE
# ==================================================

test_row = test_dataset.metadata.iloc[0]

test_image_id = test_row["image_id"]
actual_class = test_row["dx"]

test_index = test_dataset.metadata.index[
    test_dataset.metadata["image_id"]
    == test_image_id
][0]


# If the first test image isn't one of our
# selected classes, find one that is.

if actual_class not in selected_classes:

    matching_rows = test_dataset.metadata[
        test_dataset.metadata["dx"].isin(
            selected_classes
        )
    ]

    test_row = matching_rows.iloc[0]

    test_image_id = test_row[
        "image_id"
    ]

    actual_class = test_row[
        "dx"
    ]

    test_index = test_dataset.metadata.index[
        test_dataset.metadata["image_id"]
        == test_image_id
    ][0]


query_image, query_original_label = (
    test_dataset[test_index]
)

query_image = query_image.unsqueeze(
    0
).to(device)


# ==================================================
# PREDICTION
# ==================================================

with torch.no_grad():

    query_embedding = model.backbone(
        query_image
    )

    distances = torch.cdist(
        query_embedding,
        prototypes
    ).squeeze(0)

    predicted_index = torch.argmin(
        distances
    ).item()


predicted_class = selected_classes[
    predicted_index
]


# ==================================================
# DISPLAY RESULTS
# ==================================================

print(
    "\n========== QUERY IMAGE =========="
)

print(
    "Image:",
    test_image_id
)

print(
    "Actual class:",
    actual_class
)

print(
    "Predicted class:",
    predicted_class
)


print(
    "\n========== PROTOTYPE DISTANCES =========="
)

for index, class_name in enumerate(
    selected_classes
):

    print(
        f"{class_name:6s}: "
        f"{distances[index].item():.4f}"
    )


print(
    "\n========================================"
)

if predicted_class == actual_class:

    print(
        "Prediction result: CORRECT"
    )

else:

    print(
        "Prediction result: INCORRECT"
    )


print(
    "========================================"
)

print(
    "\n5-way 5-shot ProtoNet demonstration "
    "completed successfully!"
)