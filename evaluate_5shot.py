import torch

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler


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

print(
    "\nUsing device:",
    device
)


# ==================================================
# LOAD TEST DATASET
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
# EVALUATION
# ==================================================

total_correct = 0
total_queries = 0

episode_accuracies = []


print(
    "\n========== 5-SHOT TEST EVALUATION =========="
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
        # Support
        # ------------------------------------------

        support_images = []
        support_labels = []

        for index in support_indices:

            image, label = test_dataset[index]

            support_images.append(image)
            support_labels.append(label)


        # ------------------------------------------
        # Query
        # ------------------------------------------

        query_images = []
        query_labels = []

        for index in query_indices:

            image, label = test_dataset[index]

            query_images.append(image)
            query_labels.append(label)


        # ------------------------------------------
        # Tensor conversion
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
        # Model prediction
        # ------------------------------------------

        distances, predictions, prototypes = model(
            support_images,
            support_labels,
            query_images
        )


        # ------------------------------------------
        # Accuracy
        # ------------------------------------------

        correct = (
            predictions == query_labels
        ).sum().item()

        episode_accuracy = (
            correct /
            len(query_labels)
        )

        episode_accuracies.append(
            episode_accuracy
        )

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

            print(
                f"Episode "
                f"{episode_number + 1}/"
                f"{TEST_EPISODES} | "
                f"Accuracy: "
                f"{episode_accuracy * 100:.2f}%"
            )


# ==================================================
# FINAL RESULTS
# ==================================================

overall_accuracy = (
    total_correct /
    total_queries
)

average_episode_accuracy = (
    sum(episode_accuracies) /
    len(episode_accuracies)
)


print(
    "\n========== FINAL 5-SHOT TEST RESULTS =========="
)

print(
    "Total test episodes:",
    TEST_EPISODES
)

print(
    "Support samples per episode:",
    N_WAY * K_SHOT
)

print(
    "Query samples per episode:",
    N_WAY * Q_QUERY
)

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
    f"{overall_accuracy * 100:.2f}%"
)

print(
    f"Average Episode Accuracy: "
    f"{average_episode_accuracy * 100:.2f}%"
)


print(
    "\n5-way 5-shot evaluation completed successfully!"
)