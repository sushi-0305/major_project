import os
import json
import torch

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_test_transform
from utils.model import FourLayerCNN


# ==================================================
# CONFIGURATION
# ==================================================

TRAIN_FILE = "data/splits/train.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

MODEL_FILE = "prototypical_skin_model_5shot.pth"

OUTPUT_DIR = "checkpoints"

PROTOTYPE_FILE = "checkpoints/prototypes.pt"

CLASS_FILE = "checkpoints/prototype_classes.json"

IMAGE_SIZE = 128

EMBEDDING_DIM = 64

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

# Process images in batches to reduce CPU memory usage
BATCH_SIZE = 32


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nUsing device:",
    device
)


# ==================================================
# OUTPUT DIRECTORY
# ==================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==================================================
# LOAD DATASET
# ==================================================

dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(
        image_size=IMAGE_SIZE
    )
)

print(
    "Training images:",
    len(dataset)
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = FourLayerCNN(
    embedding_dim=EMBEDDING_DIM
).to(device)


checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)


# Extract ProtoNet backbone
backbone_state = {}

for key, value in checkpoint.items():

    if key.startswith("backbone."):

        new_key = key.replace(
            "backbone.",
            "",
            1
        )

        backbone_state[new_key] = value


model.load_state_dict(
    backbone_state
)

model.eval()

print(
    "5-shot CNN backbone loaded successfully!"
)


# ==================================================
# CREATE CLASS INDEX LISTS
# ==================================================

class_indices = {}

for class_name in CLASS_NAMES:

    indices = dataset.metadata[
        dataset.metadata["dx"] == class_name
    ].index.tolist()

    class_indices[class_name] = indices

    print(
        f"{class_name}: "
        f"{len(indices)} training images"
    )


# ==================================================
# GENERATE PROTOTYPES
# ==================================================

prototypes = []

print(
    "\n========== GENERATING PROTOTYPES =========="
)


with torch.no_grad():

    for class_number, class_name in enumerate(
        CLASS_NAMES
    ):

        indices = class_indices[
            class_name
        ]

        class_embeddings = []

        print(
            f"\nProcessing class "
            f"{class_number + 1}/"
            f"{len(CLASS_NAMES)}: "
            f"{class_name}"
        )


        # ------------------------------------------
        # Process images in batches
        # ------------------------------------------

        for start in range(
            0,
            len(indices),
            BATCH_SIZE
        ):

            batch_indices = indices[
                start:start + BATCH_SIZE
            ]

            images = []

            for index in batch_indices:

                image, label = dataset[index]

                images.append(
                    image
                )


            images = torch.stack(
                images
            ).to(device)


            # --------------------------------------
            # Extract embeddings
            # --------------------------------------

            embeddings = model(
                images
            )

            class_embeddings.append(
                embeddings.cpu()
            )


            processed = min(
                start + BATCH_SIZE,
                len(indices)
            )

            print(
                f"  Processed "
                f"{processed}/"
                f"{len(indices)}",
                end="\r"
            )


        # ------------------------------------------
        # Combine embeddings
        # ------------------------------------------

        class_embeddings = torch.cat(
            class_embeddings,
            dim=0
        )


        # ------------------------------------------
        # Mean embedding = prototype
        # ------------------------------------------

        prototype = class_embeddings.mean(
            dim=0
        )


        prototypes.append(
            prototype
        )


        print(
            f"\nPrototype created: "
            f"{class_name}"
        )


# ==================================================
# STACK PROTOTYPES
# ==================================================

prototypes = torch.stack(
    prototypes
)


# ==================================================
# SAVE PROTOTYPES
# ==================================================

torch.save(
    prototypes,
    PROTOTYPE_FILE
)


# ==================================================
# SAVE CLASS NAMES
# ==================================================

with open(
    CLASS_FILE,
    "w"
) as file:

    json.dump(
        CLASS_NAMES,
        file,
        indent=4
    )


# ==================================================
# FINAL INFORMATION
# ==================================================

print(
    "\n========== PROTOTYPE INFORMATION =========="
)

print(
    "Number of classes:",
    len(CLASS_NAMES)
)

print(
    "Training images used:",
    len(dataset)
)

print(
    "Prototype shape:",
    prototypes.shape
)

print(
    "\nPrototype file saved:"
)

print(
    PROTOTYPE_FILE
)

print(
    "\nClass file saved:"
)

print(
    CLASS_FILE
)

print(
    "\nFull training-set prototype generation "
    "completed successfully!"
)