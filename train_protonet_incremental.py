"""Fine-tune a CNN ProtoNet in small, validation-gated increments."""

import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils.data_augmentation import get_test_transform, get_train_transform
from utils.dataset import HAM10000Dataset
from utils.episodic_sampler import EpisodicSampler
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork


IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-validation", type=float, default=0.0)
    return parser.parse_args()


def load_episode(dataset, episode, support_count):
    support = [dataset[index] for index in episode[:support_count]]
    query = [dataset[index] for index in episode[support_count:]]
    return (
        torch.stack([image for image, _ in support]),
        torch.tensor([label for _, label in support]),
        torch.stack([image for image, _ in query]),
        torch.tensor([label for _, label in query]),
    )


def labels_for(dataset):
    return [dataset.class_to_idx[name] for name in dataset.metadata["dx"]]


def score(model, dataset, labels, episodes, seed):
    random.seed(seed)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for episode in EpisodicSampler(labels, 5, 5, 5, episodes):
            support_images, support_labels, query_images, query_labels = load_episode(
                dataset, episode, support_count=25
            )
            _, predictions, _ = model(
                support_images, support_labels, query_images
            )
            correct += (predictions == query_labels).sum().item()
            total += query_labels.numel()
    return correct / total, correct, total


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_dataset = HAM10000Dataset(
        "data/splits/train.csv", IMAGE_DIRS, get_train_transform(image_size=128)
    )
    validation_dataset = HAM10000Dataset(
        "data/splits/val.csv", IMAGE_DIRS, get_test_transform(image_size=128)
    )
    train_labels = labels_for(train_dataset)
    validation_labels = labels_for(validation_dataset)
    model = PrototypicalNetwork(FourLayerCNN(embedding_dim=64))
    model.load_state_dict(torch.load(args.resume, map_location="cpu"))
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    correct = total = 0

    for episode in EpisodicSampler(
        train_labels, args.n_way, args.k_shot, args.q_query, args.episodes
    ):
        support_images, support_labels, query_images, query_labels = load_episode(
            train_dataset, episode, support_count=args.n_way * args.k_shot
        )
        distances, predictions, _ = model(
            support_images, support_labels, query_images
        )
        targets = torch.searchsorted(torch.unique(support_labels), query_labels)
        loss = criterion(-distances, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        correct += (predictions == query_labels).sum().item()
        total += query_labels.numel()

    validation_accuracy, validation_correct, validation_total = score(
        model,
        validation_dataset,
        validation_labels,
        args.validation_episodes,
        seed=42,
    )
    print(f"train_accuracy={correct / total:.4f}")
    print(
        "validation_accuracy="
        f"{validation_accuracy:.4f} ({validation_correct}/{validation_total})"
    )
    if validation_accuracy > args.min_validation:
        torch.save(model.state_dict(), args.output)
        print(f"retained={args.output}")
    else:
        print("rejected=validation threshold not exceeded")


if __name__ == "__main__":
    main()
