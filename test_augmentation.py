from PIL import Image

from utils.data_augmentation import (
    get_train_transform,
    get_test_transform
)


# --------------------------------------------------
# Load a sample image
# --------------------------------------------------

image_path = (
    "data/HAM10000_images_part_1/"
    "ISIC_0027419.jpg"
)


print("\n========== AUGMENTATION TEST ==========")

print("Loading image...")

image = Image.open(
    image_path
).convert("RGB")


print("Original image size:", image.size)


# --------------------------------------------------
# Training transformation
# --------------------------------------------------

train_transform = get_train_transform(
    image_size=128
)

train_image = train_transform(image)


print("\nTraining transformation:")

print(
    "Shape:",
    train_image.shape
)

print(
    "Type:",
    type(train_image)
)


# --------------------------------------------------
# Testing transformation
# --------------------------------------------------

test_transform = get_test_transform(
    image_size=128
)

test_image = test_transform(image)


print("\nTesting transformation:")

print(
    "Shape:",
    test_image.shape
)

print(
    "Type:",
    type(test_image)
)


# --------------------------------------------------
# Final check
# --------------------------------------------------

if train_image.shape == (3, 128, 128):

    print(
        "\nTraining transformation: PASS"
    )

else:

    print(
        "\nTraining transformation: FAIL"
    )


if test_image.shape == (3, 128, 128):

    print(
        "Testing transformation: PASS"
    )

else:

    print(
        "Testing transformation: FAIL"
    )


print(
    "\nAugmentation test completed successfully!"
)