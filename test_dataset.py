from utils.dataset import HAM10000Dataset

# HAM10000 paths

metadata_path = r"C:\Users\malak\HAM10000\HAM10000_metadata.csv"

image_dirs = [
    r"C:\Users\malak\HAM10000\HAM10000_images_part_1",
    r"C:\Users\malak\HAM10000\HAM10000_images_part_2"
]

# Create dataset

dataset = HAM10000Dataset(
    metadata_file=metadata_path,
    image_dirs=image_dirs
)

# Basic information

print("Total images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)

# Load first image

image, label = dataset[0]

print("First image size:", image.size)
print("First image label:", label)

print("\nDataset loaded successfully!")