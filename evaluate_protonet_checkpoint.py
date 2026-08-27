"""Reproducibly evaluate a CNN ProtoNet checkpoint in 5-way/5-shot episodes."""

import argparse
import random

import torch

from utils.data_augmentation import get_test_transform
from utils.dataset import HAM10000Dataset
from utils.episodic_sampler import EpisodicSampler
from utils.model import FourLayerCNN
from utils.prototypical_network import PrototypicalNetwork


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = HAM10000Dataset(
        f"data/splits/{args.split}.csv",
        ["data/HAM10000_images_part_1", "data/HAM10000_images_part_2"],
        get_test_transform(image_size=128),
    )
    labels = [dataset.class_to_idx[name] for name in dataset.metadata["dx"]]
    episodes = list(EpisodicSampler(labels, 5, 5, 5, args.episodes))
    end = args.end if args.end is not None else args.episodes

    model = PrototypicalNetwork(FourLayerCNN(embedding_dim=64))
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()

    correct = total = 0
    with torch.no_grad():
        for episode in episodes[args.start:end]:
            support = [dataset[index] for index in episode[:25]]
            query = [dataset[index] for index in episode[25:]]
            _, predictions, _ = model(
                torch.stack([image for image, _ in support]),
                torch.tensor([label for _, label in support]),
                torch.stack([image for image, _ in query]),
            )
            targets = torch.tensor([label for _, label in query])
            correct += (predictions == targets).sum().item()
            total += targets.numel()

    print(f"correct={correct} total={total} accuracy={correct / total:.6f}")


if __name__ == "__main__":
    main()
