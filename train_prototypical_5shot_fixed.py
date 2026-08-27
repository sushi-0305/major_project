import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_train_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler


TRAIN_FILE = "data/splits/train.csv"
IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

N_WAY = 5
K_SHOT = 5
Q_QUERY = 5

EPISODES = 100
EPOCHS = 15

# Lowered 10x from the original 0.001 - that LR combined with no grad
# clipping was collapsing embeddings to a single point within ~1 epoch.
LEARNING_RATE = 0.0001

EMBEDDING_DIM = 64

MODEL_PATH = "checkpoints/prototypical_skin_model_5shot_fixed.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_train_transform(image_size=128)
)
print("Training images:", len(dataset))

labels = [dataset.class_to_idx[disease] for disease in dataset.metadata["dx"]]

backbone = FourLayerCNN(embedding_dim=EMBEDDING_DIM)
model = PrototypicalNetwork(backbone).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# Decay LR over training so late episodes don't destabilize what's learned.
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("\n========== 5-SHOT PROTOTYPICAL TRAINING (fixed) ==========")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_queries = 0

    sampler = EpisodicSampler(
        labels=labels, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY, episodes=EPISODES
    )

    for episode_number, episode in enumerate(sampler):
        support_count = N_WAY * K_SHOT
        support_indices = episode[:support_count]
        query_indices = episode[support_count:]

        support_images, support_labels = [], []
        for index in support_indices:
            image, label = dataset[index]
            support_images.append(image)
            support_labels.append(label)

        query_images, query_labels = [], []
        for index in query_indices:
            image, label = dataset[index]
            query_images.append(image)
            query_labels.append(label)

        support_images = torch.stack(support_images).to(device)
        support_labels = torch.tensor(support_labels, dtype=torch.long).to(device)
        query_images = torch.stack(query_images).to(device)
        query_labels = torch.tensor(query_labels, dtype=torch.long).to(device)

        distances, predictions, prototypes = model(support_images, support_labels, query_images)
        logits = -distances

        classes = torch.unique(support_labels)
        target_positions = torch.zeros_like(query_labels)
        for position, cls in enumerate(classes):
            target_positions[query_labels == cls] = position

        loss = nn.CrossEntropyLoss()(logits, target_positions)

        optimizer.zero_grad()
        loss.backward()
        # This was missing in the original script and is the main reason
        # embeddings were collapsing to a single point (loss = ln(5)).
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (predictions == query_labels).sum().item()
        total_queries += len(query_labels)

        if (episode_number + 1) % 20 == 0:
            print(
                f"Epoch {epoch + 1}/{EPOCHS} | Episode {episode_number + 1}/{EPISODES} | "
                f"Loss: {loss.item():.4f}"
            )

    scheduler.step()
    average_loss = total_loss / EPISODES
    accuracy = total_correct / total_queries

    print(f"\nEpoch {epoch + 1} completed")
    print(f"Average Loss: {average_loss:.4f}")
    print(f"Episode Accuracy: {accuracy:.4f}")
    print(f"Episode Accuracy (%): {accuracy * 100:.2f}%")
    print(f"LR: {scheduler.get_last_lr()[0]:.6f}")

import os
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel saved successfully: {MODEL_PATH}")
print("\n5-way 5-shot training completed successfully!")