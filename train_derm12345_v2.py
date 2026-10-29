
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.derm12345_dataset import Derm12345Dataset
from utils.data_augmentation import get_train_transform, get_test_transform
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork
from utils.episodic_sampler import EpisodicSampler

SEED = 42
N_WAY, K_SHOT, Q_QUERY = 5, 5, 5
EPISODES, VAL_EPISODES, EPOCHS = 30, 30, 15
LR, EMBEDDING_DIM, IMAGE_SIZE = 0.0001, 128, 128

TRAIN_FILE = "data/derm12345/splits/train.csv"
VAL_FILE = "data/derm12345/splits/val.csv"
CHECKPOINT = "checkpoints/derm12345_protonet_v2_best.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print("Device:", device, flush=True)

os.makedirs("checkpoints", exist_ok=True)

train_ds = Derm12345Dataset(
    TRAIN_FILE, transform=get_train_transform(image_size=IMAGE_SIZE)
)
val_ds = Derm12345Dataset(
    VAL_FILE, transform=get_test_transform(image_size=IMAGE_SIZE)
)

train_labels = [train_ds.class_to_idx[x] for x in train_ds.metadata["label"]]
val_labels = [val_ds.class_to_idx[x] for x in val_ds.metadata["label"]]

print("Training images:", len(train_ds), flush=True)
print("Validation images:", len(val_ds), flush=True)
print("Classes:", train_ds.classes, flush=True)

model = PrototypicalNetwork(
    FourLayerCNN(embedding_dim=EMBEDDING_DIM)
).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
best_val = -1.0


def make_episode(dataset, indices):
    support = [dataset[i] for i in indices[:N_WAY * K_SHOT]]
    query = [dataset[i] for i in indices[N_WAY * K_SHOT:]]

    sx = torch.stack([x for x, _ in support]).to(device)
    sy = torch.tensor([y for _, y in support], dtype=torch.long, device=device)
    qx = torch.stack([x for x, _ in query]).to(device)
    qy = torch.tensor([y for _, y in query], dtype=torch.long, device=device)
    return sx, sy, qx, qy


def episode_targets(support_labels, query_labels):
    classes = torch.unique(support_labels)
    targets = torch.zeros_like(query_labels)
    for position, label in enumerate(classes):
        targets[query_labels == label] = position
    return classes, targets


def evaluate():
    model.eval()
    random.seed(SEED + 100)
    correct = total = 0
    sampler = EpisodicSampler(
        val_labels, N_WAY, K_SHOT, Q_QUERY, VAL_EPISODES
    )

    with torch.no_grad():
        for indices in sampler:
            sx, sy, qx, qy = make_episode(val_ds, indices)
            distances, _, _ = model(sx, sy, qx)
            classes, _ = episode_targets(sy, qy)
            predictions = classes[(-distances).argmax(dim=1)]
            correct += (predictions == qy).sum().item()
            total += qy.numel()

    return correct / total if total else 0.0


print("Starting CPU pilot training...", flush=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = total_correct = total_queries = 0
    sampler = EpisodicSampler(
        train_labels, N_WAY, K_SHOT, Q_QUERY, EPISODES
    )

    for episode_num, indices in enumerate(sampler, start=1):
        sx, sy, qx, qy = make_episode(train_ds, indices)
        distances, _, _ = model(sx, sy, qx)
        classes, targets = episode_targets(sy, qy)
        loss = loss_fn(-distances, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            predictions = classes[(-distances).argmax(dim=1)]
            total_correct += (predictions == qy).sum().item()
            total_queries += qy.numel()
        total_loss += loss.item()

        print(
            f"  Epoch {epoch + 1}/{EPOCHS}, "
            f"episode {episode_num}/{EPISODES}",
            flush=True,
        )

    val_acc = evaluate()
    train_acc = total_correct / max(total_queries, 1)
    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"loss={total_loss / EPISODES:.4f} | "
        f"train_acc={train_acc:.4f} | val_acc={val_acc:.4f}",
        flush=True,
    )

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), CHECKPOINT)
        print("Saved best checkpoint:", CHECKPOINT, flush=True)

print(f"Pilot complete. Best validation accuracy: {best_val:.2%}", flush=True)
