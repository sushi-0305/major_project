from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "derm12345" / "mapped_manifest.csv"
OUT_DIR = ROOT / "data" / "derm12345" / "splits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

# Remove records without a patient ID to ensure leakage-safe splits.
df = df.dropna(subset=["patient_id"]).copy()

# Split patients: 70% train, 15% validation, 15% test.
first = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(
    first.split(df, groups=df["patient_id"])
)
train = df.iloc[train_idx].copy()
temp = df.iloc[temp_idx].copy()

second = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(
    second.split(temp, groups=temp["patient_id"])
)
val = temp.iloc[val_idx].copy()
test = temp.iloc[test_idx].copy()

for name, part in [
    ("train", train),
    ("val", val),
    ("test", test),
]:
    part.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"\n{name}: {len(part)} images, "
          f"{part['patient_id'].nunique()} patients")
    print(part["label"].value_counts().sort_index().to_string())

# Verify that no patient appears in multiple splits.
a, b, c = map(
    lambda x: set(x["patient_id"]),
    [train, val, test]
)
assert not (a & b or a & c or b & c), "Patient leakage detected!"

print("\nVerified: no patient overlaps between splits.")
print(f"Saved splits to: {OUT_DIR}")