import torch
import numpy as np
import random

# ==================================================
# REPRODUCIBILITY
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Random seed:", SEED)

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler
from utils.evaluation import calculate_metrics, print_metrics


# ==================================================
# CONFIGURATION
# ==================================================

TEST_FILE = "data/splits/test.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

MODEL_FILE = "prototypical_skin_model_5shot.pth"

N_WAY = 5
K_SHOT = 5
Q_QUERY = 5

TEST_EPISODES = 100

EMBEDDING_DIM = 64


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# ==================================================
# LOAD DATASET
# ==================================================

test_dataset = HAM10000Dataset(
    metadata_file=TEST_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(
        image_size=128
    )
)

print(
    "Test images:",
    len(test_dataset)
)


# ==================================================
# LABELS
# ==================================================

labels = [
    test_dataset.class_to_idx[disease]
    for disease in test_dataset.metadata["dx"]
]


# ==================================================
# CREATE MODEL
# ==================================================

backbone = FourLayerCNN(
    embedding_dim=EMBEDDING_DIM
)

model = PrototypicalNetwork(
    backbone
).to(device)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device
    )
)

model.eval()

print(
    "\nTrained 5-shot model loaded successfully!"
)


# ==================================================
# CREATE TEST EPISODES
# ==================================================

sampler = EpisodicSampler(
    labels=labels,
    n_way=N_WAY,
    k_shot=K_SHOT,
    q_query=Q_QUERY,
    episodes=TEST_EPISODES
)


# ==================================================
# COLLECT PREDICTIONS
# ==================================================

true_labels = []
predicted_labels = []

total_correct = 0
total_queries = 0


print(
    "\n========== 5-SHOT METRICS EVALUATION =========="
)


with torch.no_grad():

    for episode_number, episode in enumerate(
        sampler
    ):

        # ------------------------------------------
        # Separate support and query
        # ------------------------------------------

        support_count = N_WAY * K_SHOT

        support_indices = episode[
            :support_count
        ]

        query_indices = episode[
            support_count:
        ]


        # ------------------------------------------
        # Support data
        # ------------------------------------------

        support_images = []
        support_labels = []

        for index in support_indices:

            image, label = test_dataset[index]

            support_images.append(image)
            support_labels.append(label)


        # ------------------------------------------
        # Query data
        # ------------------------------------------

        query_images = []
        query_labels = []

        for index in query_indices:

            image, label = test_dataset[index]

            query_images.append(image)
            query_labels.append(label)


        # ------------------------------------------
        # Convert to tensors
        # ------------------------------------------

        support_images = torch.stack(
            support_images
        ).to(device)

        support_labels = torch.tensor(
            support_labels,
            dtype=torch.long
        ).to(device)

        query_images = torch.stack(
            query_images
        ).to(device)

        query_labels = torch.tensor(
            query_labels,
            dtype=torch.long
        ).to(device)


        # ------------------------------------------
        # ProtoNet prediction
        # ------------------------------------------

        distances, predictions, prototypes = model(
            support_images,
            support_labels,
            query_images
        )


        # ------------------------------------------
        # Store results
        # ------------------------------------------

        true_labels.extend(
            query_labels.cpu().numpy().tolist()
        )

        predicted_labels.extend(
            predictions.cpu().numpy().tolist()
        )


        correct = (
            predictions == query_labels
        ).sum().item()

        total_correct += correct

        total_queries += len(
            query_labels
        )


        # ------------------------------------------
        # Progress
        # ------------------------------------------

        if (
            episode_number + 1
        ) % 20 == 0:

            episode_accuracy = (
                correct /
                len(query_labels)
            )

            print(
                f"Episode "
                f"{episode_number + 1}/"
                f"{TEST_EPISODES} | "
                f"Accuracy: "
                f"{episode_accuracy * 100:.2f}%"
            )


# ==================================================
# CALCULATE METRICS
# ==================================================

metrics = calculate_metrics(
    true_labels,
    predicted_labels
)


# ==================================================
# PRINT METRICS
# ==================================================

print_metrics(metrics)


# ==================================================
# EXTRA INFORMATION
# ==================================================

print(
    "Total query images:",
    total_queries
)

print(
    "Correct predictions:",
    total_correct
)

print(
    f"Overall Accuracy: "
    f"{total_correct / total_queries * 100:.2f}%"
)


# ==================================================
# CLASSIFICATION REPORT
# ==================================================

class_names = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

from sklearn.metrics import classification_report

print(
    "\n========== CLASSIFICATION REPORT =========="
)

print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(7)),
        target_names=class_names,
        zero_division=0
    )
)

print(
    "5-way 5-shot metrics evaluation completed!"
)