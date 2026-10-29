import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Derm12345Dataset(Dataset):

    def __init__(self, metadata_file, transform=None, class_to_idx=None):
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform

        self.classes = ["bcc", "bkl", "df", "mel", "nv"]

        if class_to_idx is None:
            self.class_to_idx = {
                name: idx for idx, name in enumerate(self.classes)
            }
        else:
            self.class_to_idx = class_to_idx
            self.classes = [
                name for name, idx in
                sorted(class_to_idx.items(), key=lambda item: item[1])
            ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Use the full image path already saved in the manifest.
        image = Image.open(row["image_path"]).convert("RGB")
        label = self.class_to_idx[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label