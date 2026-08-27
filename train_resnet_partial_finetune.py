"""
Increment 1: Fine-tune a ResNet-18 Prototypical Network with layer4 unfrozen.

Key changes vs train_resnet_prototypical_improved.py:
  - Backbone starts frozen, then layer4 + the embedding head are unfrozen
    (everything before layer4 stays frozen -> keeps general low-level
    features, adapts only the high-level, task-specific features).
  - Differential learning rates: the unfrozen backbone gets a much smaller
    LR than the embedding head, since it's pretrained and should move
    gently.
  - More episodes per epoch and more epochs than the "incremental" script,
    so the model actually gets meaningful gradient signal.
  - Validation uses more episodes for a less noisy accuracy estimate.

Usage:
  python train_resnet_partial_finetune.py \
      --epochs 15 --episodes 150 --validation-episodes 150

Run this fresh (no --resume) to get a clean comparison point against your
46% baseline. Report back train_accuracy / validation_accuracy per epoch
and we'll decide the next increment from there.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_augmentation import get_test_transform, get_train_transform
from utils.dataset import HAM10000Dataset
from utils.episodic_sampler import EpisodicSampler
from utils.prototypical_network import PrototypicalNetwork
from utils.resnet_model import ResNet18Embedding


CLASS_COUNT = 5
SUPPORT_PER_CLASS = 5
QUERY_PER_CLASS = 5
IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--validation-episodes", type=int, default=150)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", help="Optional checkpoint to start from.")
    parser.add_argument(
        "--checkpoint", default="checkpoints/resnet_partial_finetune_best.pth"
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_labels(dataset):
    return [dataset.class_to_idx[label] for label in dataset.metadata["dx"]]


def load_episode(dataset, episode, device):
    support_count = CLASS_COUNT * SUPPORT_PER_CLASS
    support_pairs = [dataset[index] for index in episode[:support_count]]
    query_pairs = [dataset[index] for index in episode[support_count:]]
    support_images = torch.stack([image for image, _ in support_pairs]).to(device)
    support_labels = torch.tensor(
        [label for _, label in support_pairs], dtype=torch.long, device=device
    )
    query_images = torch.stack([image for image, _ in query_pairs]).to(device)
    query_labels = torch.tensor(
        [label for _, label in query_pairs], dtype=torch.long, device=device
    )
    return support_images, support_labels, query_images, query_labels


def episode_targets(support_labels, query_labels):
    classes = torch.unique(support_labels)
    return torch.searchsorted(classes, query_labels)


def evaluate(model, dataset, labels, episodes, seed, device):
    random.seed(seed)
    sampler = EpisodicSampler(
        labels,
        n_way=CLASS_COUNT,
        k_shot=SUPPORT_PER_CLASS,
        q_query=QUERY_PER_CLASS,
        episodes=episodes,
    )
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for episode in sampler:
            support_images, support_labels, query_images, query_labels = load_episode(
                dataset, episode, device
            )
            _, predictions, _ = model(support_images, support_labels, query_images)
            correct += (predictions == query_labels).sum().item()
            total += query_labels.numel()
    return correct / total


def unfreeze_layer4(backbone_module):
    """Unfreeze only layer4 (the last residual block) of a torchvision resnet18."""
    for name, param in backbone_module.named_parameters():
        if name.startswith("layer4"):
            param.requires_grad = True


def main():
    args = parse_args()
    set_seed(args.seed)
    os.environ.setdefault("TORCH_HOME", str(Path(".torch_cache").resolve()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = HAM10000Dataset(
        "data/splits/train.csv", IMAGE_DIRS, get_train_transform(image_size=128)
    )
    validation_dataset = HAM10000Dataset(
        "data/splits/val.csv", IMAGE_DIRS, get_test_transform(image_size=128)
    )
    train_labels = get_labels(train_dataset)
    validation_labels = get_labels(validation_dataset)

    backbone = ResNet18Embedding(
        embedding_dim=128, pretrained=True, freeze_backbone=True
    )
    # Selectively unfreeze layer4 of the underlying resnet18.
    unfreeze_layer4(backbone.backbone)

    model = PrototypicalNetwork(backbone).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from: {args.resume}")

    # Differential learning rates: unfrozen backbone params move gently,
    # the embedding head moves faster since it's trained from scratch.
    backbone_params = [
        p for n, p in model.backbone.backbone.named_parameters() if p.requires_grad
    ]
    head_params = list(model.backbone.embedding.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = checkpoint_path.with_suffix(".json")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Trainable params: {trainable:,} / {total:,}")
    print(f"Training episodes per epoch: {args.episodes}")

    best_accuracy = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        random.seed(args.seed + epoch)
        sampler = EpisodicSampler(
            train_labels,
            n_way=CLASS_COUNT,
            k_shot=SUPPORT_PER_CLASS,
            q_query=QUERY_PER_CLASS,
            episodes=args.episodes,
        )
        model.train()
        # Keep BatchNorm stats from drifting on the still-frozen early layers.
        # layer4 (unfrozen) will still update its own BN stats via .train().
        running_loss = correct = total_q = 0

        for episode in sampler:
            support_images, support_labels, query_images, query_labels = load_episode(
                train_dataset, episode, device
            )
            distances, predictions, _ = model(
                support_images, support_labels, query_images
            )
            loss = criterion(-distances, episode_targets(support_labels, query_labels))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            correct += (predictions == query_labels).sum().item()
            total_q += query_labels.numel()

        validation_accuracy = evaluate(
            model,
            validation_dataset,
            validation_labels,
            args.validation_episodes,
            args.seed,
            device,
        )
        scheduler.step()
        epoch_result = {
            "epoch": epoch,
            "train_loss": running_loss / args.episodes,
            "train_accuracy": correct / total_q,
            "validation_accuracy": validation_accuracy,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result))

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

        history_path.write_text(json.dumps(history, indent=2) + "\n")

    print(f"Best validation accuracy: {best_accuracy:.2%}")


if __name__ == "__main__":
    main()