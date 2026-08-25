import os

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_train_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

METADATA_FILE = "data/HAM10000_metadata.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]


# --------------------------------------------------
# Check paths
# --------------------------------------------------

print("Checking dataset paths...\n")

if not os.path.exists(METADATA_FILE):
    print("ERROR: Metadata file not found:")
    print(METADATA_FILE)
    exit()

for directory in IMAGE_DIRS:
    if os.path.exists(directory):
        print("Found:", directory)
    else:
        print("WARNING: Directory not found:")
        print(directory)


# --------------------------------------------------
# Create transformation
# --------------------------------------------------

transform = get_train_transform(image_size=128)


# --------------------------------------------------
# Create dataset
# --------------------------------------------------

dataset = HAM10000Dataset(
    metadata_file=METADATA_FILE,
    image_dirs=IMAGE_DIRS,
    transform=transform
)


# --------------------------------------------------
# Dataset information
# --------------------------------------------------

print("\n========== DATASET INFORMATION ==========")

print("Total images:", len(dataset))

# FIX:
# HAM10000Dataset uses class_to_idx instead of classes
print("Classes:", list(dataset.class_to_idx.keys()))

print("Number of classes:", len(dataset.class_to_idx))

print("Class mapping:", dataset.class_to_idx)


# --------------------------------------------------
# Test first image
# --------------------------------------------------

image, label = dataset[0]

print("\n========== FIRST IMAGE TEST ==========")

print("Image shape:", image.shape)

print("Image label:", label)

print("Image type:", type(image))

print("Label type:", type(label))


# --------------------------------------------------
# Test multiple images
# --------------------------------------------------

print("\n========== MULTIPLE IMAGE TEST ==========")

for i in range(5):
    image, label = dataset[i]

    print(
        f"Image {i}: "
        f"shape={image.shape}, "
        f"label={label}"
    )


print("\nDataset loaded successfully!")