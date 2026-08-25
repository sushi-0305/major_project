import os
import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

METADATA_FILE = "data/HAM10000_metadata.csv"

OUTPUT_DIR = "data/splits"


# --------------------------------------------------
# Create output directory
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

print("Loading HAM10000 metadata...")

df = pd.read_csv(METADATA_FILE)

print("Total images:", len(df))


# --------------------------------------------------
# Check classes
# --------------------------------------------------

print("\nClass distribution:")
print(df["dx"].value_counts())


# --------------------------------------------------
# 70% Train / 30% Temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["dx"]
)


# --------------------------------------------------
# 15% Validation / 15% Test
# --------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["dx"]
)


# --------------------------------------------------
# Save files
# --------------------------------------------------

train_path = os.path.join(
    OUTPUT_DIR,
    "train.csv"
)

val_path = os.path.join(
    OUTPUT_DIR,
    "val.csv"
)

test_path = os.path.join(
    OUTPUT_DIR,
    "test.csv"
)


train_df.to_csv(
    train_path,
    index=False
)

val_df.to_csv(
    val_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n========== DATASET SPLIT ==========")

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Testing images:", len(test_df))

print(
    "Total:",
    len(train_df) +
    len(val_df) +
    len(test_df)
)


print("\n========== TRAINING DISTRIBUTION ==========")
print(train_df["dx"].value_counts())


print("\n========== VALIDATION DISTRIBUTION ==========")
print(val_df["dx"].value_counts())


print("\n========== TEST DISTRIBUTION ==========")
print(test_df["dx"].value_counts())


print("\nDataset split completed successfully!")

print("\nFiles created:")
print(train_path)
print(val_path)
print(test_path)