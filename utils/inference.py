import torch
import os

from PIL import Image

from utils.model import FourLayerCNN
from utils.data_augmentation import get_test_transform


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_FILE = "prototypical_skin_model_5shot.pth"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

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


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==================================================
# LOAD MODEL
# ==================================================

def load_model():

    model = FourLayerCNN(
        embedding_dim=EMBEDDING_DIM
    ).to(device)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device
    )

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

    return model


# ==================================================
# IMAGE PREPROCESSING
# ==================================================

transform = get_test_transform(
    image_size=IMAGE_SIZE
)


# ==================================================
# LOAD IMAGE
# ==================================================

def load_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    )

    tensor = tensor.unsqueeze(
        0
    )

    return tensor.to(device)


# ==================================================
# EXTRACT EMBEDDING
# ==================================================

def get_embedding(
    model,
    image_tensor
):

    with torch.no_grad():

        embedding = model(
            image_tensor
        )

    return embedding


# ==================================================
# EUCLIDEAN DISTANCE
# ==================================================

def calculate_distances(
    query_embedding,
    prototypes
):

    distances = torch.cdist(
        query_embedding,
        prototypes
    )

    return distances


# ==================================================
# PREDICTION
# ==================================================

def predict(
    model,
    image_path,
    prototypes
):

    image_tensor = load_image(
        image_path
    )

    query_embedding = get_embedding(
        model,
        image_tensor
    )

    distances = calculate_distances(
        query_embedding,
        prototypes
    )

    predicted_index = torch.argmin(
        distances,
        dim=1
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    return {
        "class": predicted_class,
        "class_index": predicted_index,
        "distances": distances.cpu().numpy(),
        "embedding": query_embedding.cpu().numpy()
    }