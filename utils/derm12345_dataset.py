from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Derm12345Dataset(Dataset):
    CLASSES = ["bcc", "bkl", "df", "mel", "nv"]

    def __init__(self, manifest_file, transform=None):
        self.metadata = pd.read_csv(manifest_file)
        self.transform = transform

        self.metadata = self.metadata[
            self.metadata["label"].isin(self.CLASSES)
        ].reset_index(drop=True)

        self.class_to_idx = {
            label: index for index, label in enumerate(self.CLASSES)
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        image_path = Path(row["image_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[row["label"]]
        return image, label

    @property
    def classes(self):
        return self.CLASSES
