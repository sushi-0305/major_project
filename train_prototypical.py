import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_train_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRAIN_FILE = "data/splits/train.csv"

IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

N_WAY = 5
K_SHOT = 1
Q_QUERY = 5

EPISODES = 200
EPOCHS = 10

LEARNING_RATE = 0.0005

EMBEDDING_DIM = 128


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_train_transform(
        image_size=128
    )
)

print(
    "Training images:",
    len(dataset)
)


# --------------------------------------------------
# Labels
# --------------------------------------------------

labels = [
    dataset.class_to_idx[disease]
    for disease in dataset.metadata["dx"]
]


# --------------------------------------------------
# CNN
# --------------------------------------------------

backbone = FourLayerCNN(
    embedding_dim=EMBEDDING_DIM
)


# --------------------------------------------------
# Prototypical Network
# --------------------------------------------------

model = PrototypicalNetwork(
    backbone
).to(device)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\n========== TRAINING ==========")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_queries = 0

    sampler = EpisodicSampler(
        labels=labels,
        n_way=N_WAY,
        k_shot=K_SHOT,
        q_query=Q_QUERY,
        episodes=EPISODES
    )

    for episode_number, episode in enumerate(
        sampler
    ):

        # ------------------------------------------
        # Separate support and query indices
        # ------------------------------------------

        support_count = N_WAY * K_SHOT

        support_indices = episode[
            :support_count
        ]

        query_indices = episode[
            support_count:
        ]


        # ------------------------------------------
        # Load support images
        # ------------------------------------------

        support_images = []
        support_labels = []

        for index in support_indices:

            image, label = dataset[index]

            support_images.append(image)
            support_labels.append(label)


        # ------------------------------------------
        # Load query images
        # ------------------------------------------

        query_images = []
        query_labels = []

        for index in query_indices:

            image, label = dataset[index]

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
        # Forward pass
        # ------------------------------------------

        distances, predictions, prototypes = model(
            support_images,
            support_labels,
            query_images
        )


        # ------------------------------------------
        # Convert distances to logits
        # Smaller distance = higher probability
        # ------------------------------------------

        logits = -distances


        # ------------------------------------------
        # Get prototype class order
        # ------------------------------------------

        classes = torch.unique(
            support_labels
        )


        # ------------------------------------------
        # Convert query labels into positions
        # within the selected episode classes
        # ------------------------------------------

        target_positions = torch.zeros_like(
            query_labels
        )

        for position, cls in enumerate(classes):

            target_positions[
                query_labels == cls
            ] = position


        # ------------------------------------------
        # Calculate loss
        # ------------------------------------------

        loss = nn.CrossEntropyLoss()(
            logits,
            target_positions
        )


        # ------------------------------------------
        # Backpropagation
        # ------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        # ------------------------------------------
        # Statistics
        # ------------------------------------------

        total_loss += loss.item()

        total_correct += (
            predictions == query_labels
        ).sum().item()

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
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Episode {episode_number + 1}/{EPISODES} | "
                f"Loss: {loss.item():.4f}"
            )


    # --------------------------------------------------
    # Epoch results
    # --------------------------------------------------

    average_loss = (
        total_loss / EPISODES
    )

    accuracy = (
        total_correct /
        total_queries
    )

    print(
        f"\nEpoch {epoch + 1} completed"
    )

    print(
        f"Average Loss: {average_loss:.4f}"
    )

    print(
        f"Episode Accuracy: {accuracy:.4f}"
    )

    print(
        f"Episode Accuracy (%): "
        f"{accuracy * 100:.2f}%"
    )


# --------------------------------------------------
# Save model
# --------------------------------------------------

MODEL_PATH = "prototypical_skin_model_improved.pth"

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(
    f"\nModel saved successfully: {MODEL_PATH}"
)

print(
    "\nTraining completed successfully!"
)