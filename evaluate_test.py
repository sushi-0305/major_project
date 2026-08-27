"""Evaluate a saved Prototypical Network checkpoint on the held-out test set.

Usage:
  python evaluate_test.py --checkpoint checkpoints/prototypical_skin_model_5shot_best.pth
  python evaluate_test.py --checkpoint checkpoints/resnet_partial_finetune_best.pth --backbone resnet
"""

import argparse
import random

import torch

from utils.data_augmentation import get_test_transform
from utils.dataset import HAM10000Dataset
from utils.episodic_sampler import EpisodicSampler
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork


TEST_FILE = "data/splits/test.csv"
IMAGE_DIRS = [
    "data/HAM10000_images_part_1",
    "data/HAM10000_images_part_2",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--q-query", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def load_episode(dataset, episode, support_count, device):
    support_pairs = [dataset[i] for i in episode[:support_count]]
    query_pairs = [dataset[i] for i in episode[support_count:]]
    return (
        torch.stack([img for img, _ in support_pairs]).to(device),
        torch.tensor([lbl for _, lbl in support_pairs], dtype=torch.long).to(device),
        torch.stack([img for img, _ in query_pairs]).to(device),
        torch.tensor([lbl for _, lbl in query_pairs], dtype=torch.long).to(device),
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HAM10000Dataset(
        TEST_FILE, IMAGE_DIRS, get_test_transform(image_size=128)
    )
    labels = [dataset.class_to_idx[d] for d in dataset.metadata["dx"]]
    print(f"Test images: {len(dataset)}")

    if args.backbone == "cnn":
        backbone = FourLayerCNN(embedding_dim=args.embedding_dim)
    else:
        from utils.resnet_model import ResNet18Embedding
        backbone = ResNet18Embedding(
            embedding_dim=args.embedding_dim, pretrained=True, freeze_backbone=True
        )

    model = PrototypicalNetwork(backbone).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    random.seed(args.seed)
    sampler = EpisodicSampler(
        labels, args.n_way, args.k_shot, args.q_query, args.episodes
    )

    correct = total = 0
    per_episode_accuracies = []

    with torch.no_grad():
        for episode in sampler:
            support_images, support_labels, query_images, query_labels = load_episode(
                dataset, episode, args.n_way * args.k_shot, device
            )
            _, predictions, _ = model(support_images, support_labels, query_images)
            episode_correct = (predictions == query_labels).sum().item()
            episode_total = query_labels.numel()
            correct += episode_correct
            total += episode_total
            per_episode_accuracies.append(episode_correct / episode_total)

    accuracy = correct / total
    std = torch.tensor(per_episode_accuracies).std().item()
    # 95% confidence interval across episodes, standard for few-shot reporting.
    ci95 = 1.96 * std / (len(per_episode_accuracies) ** 0.5)

    print(f"\n========== TEST RESULTS ==========")
    print(f"Episodes: {args.episodes} ({args.n_way}-way {args.k_shot}-shot)")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"95% CI:   +/- {ci95:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()