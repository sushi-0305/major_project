import torch

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork


# --------------------------------------------------
# Paths
# --------------------------------------------------

TRAIN_FILE = "data/splits/train.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(
        image_size=128
    )
)


# --------------------------------------------------
# Select 5 classes
# --------------------------------------------------

selected_classes = [
    0,
    1,
    2,
    3,
    4
]


# --------------------------------------------------
# Select support and query samples
# --------------------------------------------------

support_images = []
support_labels = []

query_images = []
query_labels = []


for cls in selected_classes:

    class_indices = [
        i
        for i, disease
        in enumerate(dataset.metadata["dx"])
        if dataset.class_to_idx[disease] == cls
    ]

    # 1 support + 5 query
    selected_indices = class_indices[:6]

    # Support image
    image, label = dataset[
        selected_indices[0]
    ]

    support_images.append(image)
    support_labels.append(label)

    # Query images
    for index in selected_indices[1:]:

        image, label = dataset[index]

        query_images.append(image)
        query_labels.append(label)


# --------------------------------------------------
# Convert to tensors
# --------------------------------------------------

support_images = torch.stack(
    support_images
)

support_labels = torch.tensor(
    support_labels
)

query_images = torch.stack(
    query_images
)

query_labels = torch.tensor(
    query_labels
)


# --------------------------------------------------
# Print episode information
# --------------------------------------------------

print(
    "\n========== PROTOTYPICAL NETWORK TEST =========="
)

print(
    "Support images:",
    support_images.shape
)

print(
    "Support labels:",
    support_labels
)

print(
    "Query images:",
    query_images.shape
)

print(
    "Query labels:",
    query_labels
)


# --------------------------------------------------
# Create CNN
# --------------------------------------------------

backbone = FourLayerCNN(
    embedding_dim=64
)


# --------------------------------------------------
# Create Prototypical Network
# --------------------------------------------------

model = PrototypicalNetwork(
    backbone
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

with torch.no_grad():

    distances, predictions, prototypes = model(
        support_images,
        support_labels,
        query_images
    )


# --------------------------------------------------
# Print results
# --------------------------------------------------

print(
    "\n========== RESULTS =========="
)

print(
    "Distance shape:",
    distances.shape
)

print(
    "Prototype shape:",
    prototypes.shape
)

print(
    "Predictions:",
    predictions
)


# --------------------------------------------------
# Calculate accuracy
# --------------------------------------------------

accuracy = (
    predictions == query_labels
).float().mean()


print(
    "\nAccuracy:",
    accuracy.item()
)


# --------------------------------------------------
# Final checks
# --------------------------------------------------

if prototypes.shape == (5, 64):

    print(
        "Prototype test: PASS"
    )

else:

    print(
        "Prototype test: FAIL"
    )


if distances.shape == (25, 5):

    print(
        "Distance test: PASS"
    )

else:

    print(
        "Distance test: FAIL"
    )


print(
    "\nPrototypical Network test completed!"
)