from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "derm12345" / "metadata.csv"
IMAGE_DIR = ROOT / "data" / "derm12345" / "images"
OUT_PATH = ROOT / "data" / "derm12345" / "mapped_manifest.csv"

df = pd.read_csv(CSV_PATH)

# Conservative mapping: exact diagnosis_3 values only.
label_map = {
    "Nevus": "nv",
    "Melanoma, NOS": "mel",
    "Melanoma Invasive": "mel",
    "Melanoma in situ": "mel",
    "Basal cell carcinoma": "bcc",
    "Dermatofibroma": "df",
    "Seborrheic keratosis": "bkl",
}

df["label"] = df["diagnosis_3"].map(label_map)
df["image_path"] = df["isic_id"].astype(str).map(
    lambda image_id: str(IMAGE_DIR / f"{image_id}.jpg")
)

# Retain only mapped diagnoses with an existing image.
df["image_exists"] = df["image_path"].map(
    lambda path: Path(path).is_file()
)
mapped = df[df["label"].notna() & df["image_exists"]].copy()

mapped.to_csv(OUT_PATH, index=False)

print(f"Metadata rows: {len(df)}")
print(f"Mapped images saved: {len(mapped)}")
print(f"Missing mapped images: {(df['label'].notna() & ~df['image_exists']).sum()}")
print("\nClass counts:")
print(mapped["label"].value_counts().sort_index().to_string())
print(f"\nSaved: {OUT_PATH}")
print(f"Unique patients in mapped data: {mapped['patient_id'].nunique()}")