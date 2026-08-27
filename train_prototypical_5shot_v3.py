"""
Round 2 CNN training.

Round 1 result: train_acc=63.2%, val_acc=61.8% at convergence - the gap is
small, meaning the model is NOT overfitting. That means there's still
headroom: more capacity and more training time should help rather than hurt.

Changes vs the v2 script that got 62.16% val:
  - EMBEDDING_DIM 64 -> 128 (more representational capacity)
  - EPISODES 100 -> 150 per epoch (less noisy gradient estimates)
  - EPOCHS 30 -> 45 (longer cosine schedule so it has more room to keep
    improving before LR decays to ~0)
  - Slightly higher peak LR (1.5e-4) since the bigger model has more
    parameters to move
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils.dataset import HAM10000Dataset
from utils.data_augmentation import get_train_transform, get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler


TRAIN_FILE = "data/splits/train.csv"
VAL_FILE = "data/splits/val.csv"
IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2"
]

N_WAY = 5
K_SHOT = 5
Q_QUERY = 5

EPISODES = 150
VALIDATION_EPISODES = 100
EPOCHS = 45

LEARNING_RATE = 0.00015
EMBEDDING_DIM = 128

CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "prototypical_skin_model_5shot_v3_best.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:", device)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

train_dataset = HAM10000Dataset(
    metadata_file=TRAIN_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_train_transform(image_size=128)
)
val_dataset = HAM10000Dataset(
    metadata_file=VAL_FILE,
    image_dirs=IMAGE_DIRS,
    transform=get_test_transform(image_size=128)
)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

train_labels = [train_dataset.class_to_idx[d] for d in train_dataset.metadata["dx"]]
val_labels = [val_dataset.class_to_idx[d] for d in val_dataset.metadata["dx"]]

backbone = FourLayerCNN(embedding_dim=EMBEDDING_DIM)
model = PrototypicalNetwork(backbone).to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def load_episode(dataset, episode, support_count):
    support = [dataset[i] for i in episode[:support_count]]
    query = [dataset[i] for i in episode[support_count:]]
    return (
        torch.stack([img for img, _ in support]).to(device),
        torch.tensor([lbl for _, lbl in support], dtype=torch.long).to(device),
        torch.stack([img for img, _ in query]).to(device),
        torch.tensor([lbl for _, lbl in query], dtype=torch.long).to(device),
    )


def evaluate(model, dataset, labels, episodes, seed):
    random.seed(seed)
    model.eval()
    correct = total = 0
    sampler = EpisodicSampler(labels, N_WAY, K_SHOT, Q_QUERY, episodes)
    with torch.no_grad():
        for episode in sampler:
            support_images, support_labels, query_images, query_labels = load_episode(
                dataset, episode, N_WAY * K_SHOT
            )
            _, predictions, _ = model(support_images, support_labels, query_images)
            correct += (predictions == query_labels).sum().item()
            total += query_labels.numel()
    return correct / total, correct, total


print("\n========== 5-SHOT PROTOTYPICAL TRAINING (v3: bigger + longer) ==========")

best_val_accuracy = -1.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_queries = 0

    sampler = EpisodicSampler(train_labels, N_WAY, K_SHOT, Q_QUERY, EPISODES)

    for episode_number, episode in enumerate(sampler):
        support_images, support_labels, query_images, query_labels = load_episode(
            train_dataset, episode, N_WAY * K_SHOT
        )

        distances, predictions, prototypes = model(support_images, support_labels, query_images)
        logits = -distances

        classes = torch.unique(support_labels)
        target_positions = torch.zeros_like(query_labels)
        for position, cls in enumerate(classes):
            target_positions[query_labels == cls] = position

        loss = nn.CrossEntropyLoss()(logits, target_positions)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_correct += (predictions == query_labels).sum().item()
        total_queries += len(query_labels)

    scheduler.step()
    train_accuracy = total_correct / total_queries
    val_accuracy, val_correct, val_total = evaluate(
        model, val_dataset, val_labels, VALIDATION_EPISODES, seed=42
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"train_loss={total_loss / EPISODES:.4f} | "
        f"train_acc={train_accuracy:.4f} | "
        f"val_acc={val_accuracy:.4f} ({val_correct}/{val_total}) | "
        f"lr={scheduler.get_last_lr()[0]:.6f}"
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"  -> new best, saved to {BEST_MODEL_PATH}")

print(f"\nBest validation accuracy: {best_val_accuracy:.2%}")
print("Training completed successfully!")