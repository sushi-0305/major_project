import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HAM10000Dataset(Dataset):

    def __init__(
        self,
        metadata_file,
        image_dirs,
        transform=None,
        class_to_idx=None
    ):

        self.metadata = pd.read_csv(metadata_file)

        self.image_dirs = image_dirs

        self.transform = transform

        # Fixed class mapping for the entire HAM10000 dataset
        if class_to_idx is None:

            self.classes = [
                "akiec",
                "bcc",
                "bkl",
                "df",
                "mel",
                "nv",
                "vasc"
            ]

            self.class_to_idx = {
                class_name: idx
                for idx, class_name in enumerate(self.classes)
            }

        else:

            self.class_to_idx = class_to_idx

            # Get class names in index order
            self.classes = [
                class_name
                for class_name, idx
                in sorted(
                    class_to_idx.items(),
                    key=lambda x: x[1]
                )
            ]

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, idx):

        row = self.metadata.iloc[idx]

        image_id = row["image_id"]

        disease = row["dx"]

        # Find image
        image_path = None

        for directory in self.image_dirs:

            path = os.path.join(
                directory,
                image_id + ".jpg"
            )

            if os.path.exists(path):

                image_path = path

                break

        if image_path is None:

            raise FileNotFoundError(
                f"Image not found: {image_id}.jpg"
            )

        # Load image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Convert disease name to numerical label
        label = self.class_to_idx[disease]

        # Apply transformation
        if self.transform:

            image = self.transform(image)

        return image, label