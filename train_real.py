import os
import pandas as pd
import torch
from torchvision import transforms

from utils.dataset import MedicalImageDataset, FewShotDataLoader
from models.prototypical_networks import (
    PrototypicalNetworks,
    ConvNet,
    PrototypicalNetworksTrainer,
)


# -----------------------------
# Configuration
# -----------------------------
DATA_DIR = "data"
SPLIT_DIR = os.path.join(DATA_DIR, "splits")

IMAGE_DIRS = [
    os.path.join(DATA_DIR, "HAM10000_images_part_1"),
    os.path.join(DATA_DIR, "HAM10000_images_part_2"),
]

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

N_WAY = 5
N_SHOT = 5
N_QUERY = 5

EPOCHS = 5
EPISODES_PER_EPOCH = 50

IMAGE_SIZE = 84
LEARNING_RATE = 0.001

CHECKPOINT_DIR = "checkpoints/ham10000_prototypical"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# -----------------------------
# Find image path
# -----------------------------
def find_image(image_id):
    filename = image_id + ".jpg"

    for directory in IMAGE_DIRS:
        path = os.path.join(directory, filename)

        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Image not found: {image_id}")


# -----------------------------
# Load split
# -----------------------------
def load_split(filename):
    csv_path = os.path.join(SPLIT_DIR, filename)

    df = pd.read_csv(csv_path)

    image_paths = []
    labels = []

    for _, row in df.iterrows():
        image_paths.append(find_image(row["image_id"]))
        labels.append(CLASS_TO_ID[row["dx"]])

    return image_paths, labels


# -----------------------------
# Main
# -----------------------------
def main():

    device = torch.device("cpu")

    print("=" * 60)
    print("HAM10000 Prototypical Networks")
    print("=" * 60)
    print(f"Using device: {device}")

    # Transform real skin images
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load train/validation/test splits
    train_images, train_labels = load_split("train.csv")
    val_images, val_labels = load_split("val.csv")
    test_images, test_labels = load_split("test.csv")

    print(f"Training images:   {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Testing images:    {len(test_images)}")

    # Create datasets
    train_dataset = MedicalImageDataset(
        train_images,
        train_labels,
        transform=transform,
    )

    val_dataset = MedicalImageDataset(
        val_images,
        val_labels,
        transform=transform,
    )

    test_dataset = MedicalImageDataset(
        test_images,
        test_labels,
        transform=transform,
    )

    # Create ProtoNet
    backbone = ConvNet(
        in_channels=3,
        hidden_dim=64,
    )

    model = PrototypicalNetworks(
        backbone,
        hidden_dim=64,
    )

    trainer = PrototypicalNetworksTrainer(
        model,
        device,
        lr=LEARNING_RATE,
    )

    best_val_accuracy = 0.0

    # -----------------------------
    # Training
    # -----------------------------
    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loader = FewShotDataLoader(
            train_dataset,
            n_way=N_WAY,
            n_shot=N_SHOT,
            n_query=N_QUERY,
            num_episodes=EPISODES_PER_EPOCH,
        )

        # Train directly on sampled few-shot episodes
        total_loss = 0.0
        total_accuracy = 0.0

        model.train()

        for episode, (
            support_images,
            support_labels,
            query_images,
            query_labels,
        ) in enumerate(train_loader):

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            logits = model(
                support_images,
                support_labels,
                query_images,
            )

            loss = torch.nn.functional.cross_entropy(
                logits,
                query_labels,
            )

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            predictions = logits.argmax(dim=1)

            accuracy = (
                predictions == query_labels
            ).float().mean().item()

            total_loss += loss.item()
            total_accuracy += accuracy

        train_loss = total_loss / EPISODES_PER_EPOCH
        train_accuracy = total_accuracy / EPISODES_PER_EPOCH

        print(f"Train Loss:     {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()

        val_loader = FewShotDataLoader(
            val_dataset,
            n_way=N_WAY,
            n_shot=N_SHOT,
            n_query=N_QUERY,
            num_episodes=30,
        )

        val_accuracy_total = 0.0

        with torch.no_grad():

            for (
                support_images,
                support_labels,
                query_images,
                query_labels,
            ) in val_loader:

                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)

                logits = model(
                    support_images,
                    support_labels,
                    query_images,
                )

                predictions = logits.argmax(dim=1)

                accuracy = (
                    predictions == query_labels
                ).float().mean().item()

                val_accuracy_total += accuracy

        val_accuracy = val_accuracy_total / len(val_loader)

        print(f"Validation Accuracy: {val_accuracy:.4f}")

        # Save best model
        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "proto_networks_best.pth",
            )

            torch.save(
                model.state_dict(),
                checkpoint_path,
            )

            print("✓ Best model saved")


    # -----------------------------
    # Test
    # -----------------------------
    print("\n" + "=" * 60)
    print("FINAL TEST")
    print("=" * 60)

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        "proto_networks_best.pth",
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device,
        )
    )

    model.eval()

    test_loader = FewShotDataLoader(
        test_dataset,
        n_way=N_WAY,
        n_shot=N_SHOT,
        n_query=N_QUERY,
        num_episodes=5,
    )

    test_accuracy_total = 0.0

    with torch.no_grad():

        for (
            support_images,
            support_labels,
            query_images,
            query_labels,
        ) in test_loader:

            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)

            logits = model(
                support_images,
                support_labels,
                query_images,
            )

            predictions = logits.argmax(dim=1)

            accuracy = (
                predictions == query_labels
            ).float().mean().item()

            test_accuracy_total += accuracy

    test_accuracy = test_accuracy_total / len(test_loader)

    print(f"\nFINAL TEST ACCURACY: {test_accuracy:.4f}")
    print(f"FINAL TEST ACCURACY: {test_accuracy * 100:.2f}%")

    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()
