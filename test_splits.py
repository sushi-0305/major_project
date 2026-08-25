from utils.dataset import HAM10000Dataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]


TRAIN_FILE = "data/splits/train.csv"

VAL_FILE = "data/splits/val.csv"

TEST_FILE = "data/splits/test.csv"


# --------------------------------------------------
# Create datasets
# --------------------------------------------------

train_dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS
)

val_dataset = HAM10000Dataset(
    metadata_file=VAL_FILE,
    image_dirs=IMAGE_DIRS
)

test_dataset = HAM10000Dataset(
    metadata_file=TEST_FILE,
    image_dirs=IMAGE_DIRS
)


# --------------------------------------------------
# Print information
# --------------------------------------------------

print("\n========== DATASET SPLITS ==========")

print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Testing images:",
    len(test_dataset)
)


print("\n========== CLASSES ==========")

print(
    "Train classes:",
    train_dataset.classes
)

print(
    "Validation classes:",
    val_dataset.classes
)

print(
    "Test classes:",
    test_dataset.classes
)


print("\n========== CLASS MAPPING ==========")

print(
    "Train mapping:",
    train_dataset.class_to_idx
)

print(
    "Validation mapping:",
    val_dataset.class_to_idx
)

print(
    "Test mapping:",
    test_dataset.class_to_idx
)


# --------------------------------------------------
# Test images
# --------------------------------------------------

print("\n========== IMAGE TEST ==========")

train_image, train_label = train_dataset[0]

val_image, val_label = val_dataset[0]

test_image, test_label = test_dataset[0]


print(
    "Train first image:",
    train_image.size
)

print(
    "Train first label:",
    train_label
)

print(
    "Validation first image:",
    val_image.size
)

print(
    "Validation first label:",
    val_label
)

print(
    "Test first image:",
    test_image.size
)

print(
    "Test first label:",
    test_label
)


print("\nAll dataset splits loaded successfully!")