import torch

from utils.model import FourLayerCNN
from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_test_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

TEST_FILE = "data/splits/test.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

dataset = HAM10000Dataset(
    metadata_file=TEST_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(
        image_size=128
    )
)


# --------------------------------------------------
# Get one image
# --------------------------------------------------

image, label = dataset[0]

print("\n========== MODEL TEST ==========")

print(
    "Image shape:",
    image.shape
)

print(
    "Label:",
    label
)


# --------------------------------------------------
# Add batch dimension
# --------------------------------------------------

image = image.unsqueeze(0)

print(
    "Batch shape:",
    image.shape
)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = FourLayerCNN(
    embedding_dim=64
)


print(
    "\nModel created successfully!"
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

with torch.no_grad():

    embedding = model(image)


# --------------------------------------------------
# Print embedding
# --------------------------------------------------

print(
    "\nEmbedding shape:",
    embedding.shape
)

print(
    "Expected shape:",
    torch.Size([1, 64])
)


# --------------------------------------------------
# Final check
# --------------------------------------------------

if embedding.shape == (1, 64):

    print(
        "\n4-layer CNN test: PASS"
    )

else:

    print(
        "\n4-layer CNN test: FAIL"
    )


print(
    "\nModel test completed successfully!"
)