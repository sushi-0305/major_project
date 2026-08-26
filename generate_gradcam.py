import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from utils.model import FourLayerCNN


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_FILE = "prototypical_skin_model_5shot.pth"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

OUTPUT_DIR = "results/gradcam"

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

print(
    "\nUsing device:",
    device
)


# ==================================================
# FIND SAMPLE IMAGES
# ==================================================

metadata_file = "data/splits/test.csv"

import pandas as pd

metadata = pd.read_csv(
    metadata_file
)

# Select one sample from different classes
selected_rows = []

for class_name in CLASS_NAMES:

    rows = metadata[
        metadata["dx"] == class_name
    ]

    if len(rows) > 0:

        selected_rows.append(
            rows.iloc[0]
        )


# ==================================================
# CREATE MODEL
# ==================================================

model = FourLayerCNN(
    embedding_dim=EMBEDDING_DIM
).to(device)


# ==================================================
# LOAD TRAINED BACKBONE
# ==================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=device
)

# The ProtoNet checkpoint contains keys such as:
# backbone.encoder...
# backbone.embedding...

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
    "\n5-shot CNN backbone loaded successfully!"
)


# ==================================================
# TARGET LAYER
# ==================================================

target_layer = model.encoder[-3]


# ==================================================
# GRAD-CAM STORAGE
# ==================================================

activations = None
gradients = None


def forward_hook(
    module,
    input,
    output
):

    global activations

    activations = output


def backward_hook(
    module,
    grad_input,
    grad_output
):

    global gradients

    gradients = grad_output[0]


forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ==================================================
# TRANSFORM
# ==================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==================================================
# GRAD-CAM FUNCTION
# ==================================================

def generate_gradcam(
    image_path,
    output_path,
    title
):

    global activations
    global gradients

    # ----------------------------------------------
    # Load image
    # ----------------------------------------------

    original_image = Image.open(
        image_path
    ).convert("RGB")

    original_array = np.array(
        original_image
    )


    # ----------------------------------------------
    # Prepare tensor
    # ----------------------------------------------

    image_tensor = transform(
        original_image
    ).unsqueeze(0).to(device)


    # ----------------------------------------------
    # Forward pass
    # ----------------------------------------------

    model.zero_grad()

    embedding = model(
        image_tensor
    )


    # ----------------------------------------------
    # Create a scalar target
    #
    # We use the embedding magnitude as the
    # visualization target for the learned
    # feature representation.
    # ----------------------------------------------

    target = embedding.norm(
        p=2
    )


    # ----------------------------------------------
    # Backward pass
    # ----------------------------------------------

    target.backward()


    # ----------------------------------------------
    # Extract activations and gradients
    # ----------------------------------------------

    acts = activations.detach()
    grads = gradients.detach()


    # ----------------------------------------------
    # Calculate channel weights
    # ----------------------------------------------

    weights = grads.mean(
        dim=(2, 3),
        keepdim=True
    )


    # ----------------------------------------------
    # Weighted feature maps
    # ----------------------------------------------

    cam = (
        weights * acts
    ).sum(
        dim=1
    )


    # ----------------------------------------------
    # ReLU
    # ----------------------------------------------

    cam = torch.relu(
        cam
    )


    # ----------------------------------------------
    # Normalize
    # ----------------------------------------------

    cam = cam.squeeze().cpu().numpy()

    if cam.max() > 0:

        cam = (
            cam - cam.min()
        ) / (
            cam.max() - cam.min()
        )


    # ----------------------------------------------
    # Resize heatmap
    # ----------------------------------------------

    heatmap = Image.fromarray(
        np.uint8(
            cam * 255
        )
    )

    heatmap = heatmap.resize(
        (
            original_image.width,
            original_image.height
        )
    )

    heatmap_array = np.array(
        heatmap
    )


    # ----------------------------------------------
    # Create visualization
    # ----------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5)
    )


    axes[0].imshow(
        original_array
    )

    axes[0].set_title(
        "Original Image"
    )

    axes[0].axis(
        "off"
    )


    axes[1].imshow(
        original_array
    )

    axes[1].imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45
    )

    axes[1].set_title(
        "Grad-CAM Feature Attention"
    )

    axes[1].axis(
        "off"
    )


    fig.suptitle(
        title
    )

    plt.tight_layout()


    # ----------------------------------------------
    # Save
    # ----------------------------------------------

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# GENERATE VISUALIZATIONS
# ==================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


for index, row in enumerate(
    selected_rows
):

    image_id = row["image_id"]

    disease = row["dx"]

    image_path = None

    for directory in IMAGE_DIRS:

        candidate = os.path.join(
            directory,
            image_id + ".jpg"
        )

        if os.path.exists(candidate):

            image_path = candidate

            break


    if image_path is None:

        print(
            "Image not found:",
            image_id
        )

        continue


    output_path = os.path.join(
        OUTPUT_DIR,
        f"gradcam_{index + 1}_{disease}.png"
    )


    generate_gradcam(
        image_path,
        output_path,
        f"Class: {disease}"
    )


    print(
        "Generated:",
        output_path
    )


# ==================================================
# CLEANUP
# ==================================================

forward_handle.remove()
backward_handle.remove()


print(
    "\nGrad-CAM generation completed successfully!"
)